import torch
import torch.nn.functional as F
import math
from .utils import expand_dims
import numpy as np


class EVODiff:
    """
    EVODiff: Entropy-aware variance optimization for diffusion model inference 
    """
    
    def __init__(
        self,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
    ):
        
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.algorithm_type == "data_prediction":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == "edm":
            rho = 7.0  # 7.0 is the value used in the paper
            sigma_min: float = t_0
            sigma_max: float = t_T
            ramp = np.linspace(0, 1, N + 1)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            lambdas = torch.Tensor(-np.log(sigmas)).to(device)
            t = self.noise_schedule.inverse_lambda(lambdas)
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        if order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(
                    torch.tensor([0,] + orders), 0,
                ).to(device)
            ]
        return timesteps_outer, orders
        
    def compute_dot_product(self, a, b):
        """
        Compute the dot product of tensor channel and spatial dimensions (C,H,W) in batch mode.
        
        Args:
            a: First input tensor of shape (B,C,H,W)
            b: Second input tensor of shape (B,C,H,W)
                
        Returns:
            Dot product tensor with shape (B,C,1,1)
        """
        return torch.einsum('bchw,bchw->bc', a, b).view(a.size(0), -1, 1, 1)
        
    def projection_coefficient(self, tensor_a, tensor_b, tensor_b_dot_product=None):
        """
        Compute the projection coefficient for tensor_a onto tensor_b.
        
        This function calculates the coefficient that represents how much of tensor_a
        is in the direction of tensor_b. Mathematically, it's:
        projection_coefficient = (A·B) / (B·B)
        
        Parameters:
            tensor_a: The tensor being projected
            tensor_b: The tensor defining the direction
            tensor_b_dot_product: Optional pre-computed B·B to avoid recomputation
            
        Returns:
            coefficient: The scalar projection coefficient
        """
        cross_corr = self.compute_dot_product(tensor_a, tensor_b)
        if tensor_b_dot_product is None:
            var_b = self.compute_dot_product(tensor_b, tensor_b)
        else:
            var_b = tensor_b_dot_product
        return torch.clamp(cross_corr / var_b, min=-2, max=2)
        
    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)
        
    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise.

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt
 
    def sample(
        self,
        model_fn,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="evodiff",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
    ):
        
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in [
                "multistep",
            ], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in [
                "multistep",
            ], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        
        ### Initialize sliding window cache dot product variables ###
        m0_dot_m0 = None  # model_prev_0 · model_prev_0
        m1_dot_m1 = None  # model_prev_1 · model_prev_1
        m0_dot_m1 = None  # model_prev_0 · model_prev_1
        
        with torch.no_grad():
            if method == "multistep":
                assert steps >= order
                
                ### Get all scheduling parameters ###
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps 
                ns = self.noise_schedule
                all_kappas = ns.marginal_kappa(timesteps)
                all_sigmas = ns.marginal_std(timesteps)   
                
                ###Initialize ###
                step = 0   
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                
                ### Initialize first `order` iterations with lower-order methods, DDIM or Euler###
                for step in range(1, order):
                    t, s = timesteps[step], t_prev_list[-1]
                    model_s = model_prev_list[-1]
                    kappa_s, kappa_t = all_kappas[step-1], all_kappas[step]
                    sigma_s, sigma_t = all_sigmas[step-1], all_sigmas[step]
                    sigm_ratio = sigma_t / sigma_s
                    
                    if self.algorithm_type == "data_prediction":
                        h = 1/kappa_t - 1/kappa_s
                        kapra = kappa_t/kappa_s
                        x = (sigm_ratio * x + sigma_t*h* model_s)
                    else:
                        raise ValueError(
                            "Unsupported algorithm_type {}, need to be 'data_prediction'".format(self.algorithm_type)
                        )
                        
                    model_prev_list.append(self.model_fn(x, t))
                    t_prev_list.append(t)
                
                ### Use gradient-based iteration (order-step multistep) ###
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order

                    ### Get current step and scheduling params ##
                    current_idx = step
                    kappa_t, sigma_t = all_kappas[current_idx], all_sigmas[current_idx]
                    prev_idx_0 = step - 1  ### Index of t_prev_list[-1] 
                    prev_idx_1 = step - 2  ### Index of t_prev_list[-2] (If it exists)
                    
                    if step_order == 1:
                        s = t_prev_list[-1]
                        model_s = model_prev_list[-1]
                        kappa_s, sigma_s= all_kappas[prev_idx_0], all_sigmas[prev_idx_0]
                        sigma_ratio = sigma_t / sigma_s
                        
                        if self.algorithm_type == "data_prediction":
                            h = 1/kappa_t - 1/kappa_s
                            x_t = sigma_ratio * x + sigma_t*h* model_s
                            sigma_final = 1 - torch.pow(sigma_t*sigma_s, 0.5) 
                            if sigma_ratio < 0.5: 
                                r_dfinal = 1-0.5*torch.pow(sigma_ratio, 0.5)
                            else:
                                r_dfinal = 1-0.5*torch.pow(sigma_ratio - 0.5, 0.5)
    
                            D_finalstep = model_s -   r_dfinal*model_prev_list[-2]
                            x = (
                                sigma_final*x_t + 0.5*sigma_t * h*D_finalstep
                            )
                        else:
                            raise ValueError(
                                "Unsupported algorithm_type {}, need to be 'data_prediction'".format(self.algorithm_type)
                            )

                    elif step_order == 2:
                        x_pre = x
                        
                        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
                        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
                        kappa_prev_1, kappa_prev_0 = all_kappas[prev_idx_1], all_kappas[prev_idx_0]
                        sigma_prev_1, sigma_prev_0 = all_sigmas[prev_idx_1], all_sigmas[prev_idx_0]
                        sigma_rat0, sigma_ra01 = sigma_t/sigma_prev_0, sigma_prev_0/sigma_prev_1 
                        
                        h = 1/kappa_t - 1/kappa_prev_0  # step size
                        
                        #### Compute first-order parts (equivalent to Euler/DDIM iteration) ###
                        x_euler =  ( 
                            sigma_rat0 * x_pre + sigma_t*h* model_prev_0 
                        ) 
                        
                        if self.algorithm_type == "data_prediction":
                            
                            r_logh = torch.log(kappa_prev_1/kappa_prev_0)/torch.log(kappa_prev_0/kappa_t) ###the base r_i

                            if m0_dot_m0 is None:
                                m0_dot_m0 = self.compute_dot_product(model_prev_0, model_prev_0)  
                                m1_dot_m1 = self.compute_dot_product(model_prev_1, model_prev_1)
                                m0_dot_m1 = self.compute_dot_product(model_prev_0, model_prev_1)
                               
                            t_normalized = (t + t_0) / (t_T + t_0)  ### Normalized timestep
                            weight_t = 0.5 * (1 - t_normalized**2)  
                            balance_baser = torch.sqrt( sigma_rat0 / sigma_ra01)
                            ###  Compute balance parameters ### 
                            r_01_pc = torch.clamp(m0_dot_m1 / m1_dot_m1, min=-2, max=2)
                            r1_balance =  (1 - weight_t) * balance_baser + weight_t * r_01_pc
                             
                            D1_0 = model_prev_0 -   r1_balance * model_prev_1  ### Balance difference variance

                            ### Refining the control variance factor r_i ### 
                            temperature = 0.25
                            ri_scale = torch.sigmoid(temperature* r1_balance.abs())  
                            r_i = ri_scale * r_logh 
                            r_i = torch.clamp(r_i, min=0.25*r_logh, max=1.5*r_logh)
                            
                            ### Gradient-based second order probing step with variance control ### 
                            x = ( 
                                x_euler + 0.5*sigma_t*h/r_i*D1_0
                            )  
                             
                            ### Update model outputs list ###
                            for i in range(order - 1):
                                t_prev_list[i] = t_prev_list[i + 1]
                                model_prev_list[i] = model_prev_list[i + 1]
                            t_prev_list[-1] = t
                            model_t = self.model_fn(x, t)
                            model_prev_list[-1] = model_t

                            ### Compute new dot product ###
                            mt_dot_mt = self.compute_dot_product(model_t, model_t)
                            mt_dot_m0 = self.compute_dot_product(model_t, model_prev_0)
                            
                            r_t0_pc = torch.clamp(mt_dot_m0 / m0_dot_m0, min=-2, max=2)
                            r2_balance =  (1 - weight_t)  * balance_baser + weight_t * r_t0_pc
                            D2_0 = model_t -  r2_balance   * model_prev_0  ###Balance difference variance  
                            
                            B_pre_i_i, B_next_i_i = (1/r_logh/h*D1_0, r_logh/h*D2_0)  
                            eta_star =  0.5*self.projection_coefficient(B_next_i_i + B_pre_i_i, B_next_i_i - B_pre_i_i )
                            eta_star_abs = torch.abs(eta_star)  
                            eta = 0.5*torch.sigmoid(eta_star_abs)  #### Compute eta
                            eta_1, eta_2 = (-eta, 1-eta)  ### Compute eta_1, eta_2
                                
                            #### B_theta  ###
                            B_theta = eta_1/r_logh * D1_0 + r_logh*eta_2* D2_0
                            P_1 = ( x - x_euler - 0.5*sigma_t*h /r_i*B_theta )
                            D1_0_dot_D1_0 = self.compute_dot_product(D1_0, D1_0)
                            zeta_star = self.projection_coefficient(P_1, D1_0, D1_0_dot_D1_0)/(h*sigma_t)
                            zeta_star_abs = torch.abs(zeta_star)  ### Compute zeta_star_abs
                            shift_mu  = 0.5
                            zeta_star_shift =  zeta_star_abs - shift_mu 
                            
                            if steps > 20:   
                                ###  
                                D10 = (model_prev_0 - model_prev_1)/h 
                                D10_dot_D10 = self.compute_dot_product(D10, D10)
                                m0_D10_pc = self.projection_coefficient(model_prev_0, D10, D10_dot_D10)
                                consistency = torch.sigmoid(m0_D10_pc*(2 - sigma_rat0**2)) 
                                zeta = torch.sigmoid(-zeta_star_shift * consistency)
                            else:
                                zeta = torch.sigmoid( zeta_star_shift )

                            x = (
                                x_euler + 0.5*sigma_t*h/zeta * B_theta
                             )
                            
                            ### Update cached dot product values to avoid redundant computation ###
                            m1_dot_m1 = m0_dot_m0
                            m0_dot_m0 = mt_dot_mt
                            m0_dot_m1 = mt_dot_m0 ### This will be the new m0 · m1 in next step  
                            
            else:
                raise ValueError("Got wrong method {}".format(method))
            
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        
        if return_intermediate:
            return x, intermediates
        else:
            return x
            
