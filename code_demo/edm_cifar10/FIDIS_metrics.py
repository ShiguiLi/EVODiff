import tensorflow_gan as tfgan
import tensorflow as tf
import numpy as np
import os
from evaluation import *
import gc
from tqdm import tqdm

inception_model = get_inception_model(inceptionv3=False)
# BATCH_SIZE = 1000
BATCH_SIZE = 500 #(GPU,12GB)

def load_cifar10_stats():
    """Load the pre-computed dataset statistics."""
    filename = "assets/stats/cifar10_stats.npz"
    with tf.io.gfile.GFile(filename, "rb") as fin:
        stats = np.load(fin)
        return stats

def compute_metrics(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(".npz"):
            with tf.io.gfile.GFile(os.path.join(path, file), "rb") as fin:
                sample = np.load(fin)
        images.append(sample["samples"])
    samples = np.concatenate(images, axis=0)
    all_logits = []
    all_pools = []
    N = samples.shape[0]
    assert N >= 50000, "At least 50k samples are required to compute FID."
    
    for i in tqdm(range(N // BATCH_SIZE)):
        gc.collect()
        latents = run_inception_distributed(
            samples[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, ...], inception_model, inceptionv3=False
        )
        gc.collect()
        all_pools.append(latents["pool_3"])
        all_logits.append(latents["logits"])  # Store logits for IS calculation
    
    all_pools = np.concatenate(all_pools, axis=0)[:50000, ...]
    all_logits = np.concatenate(all_logits, axis=0)[:50000, ...]
    
    # Calculate FID
    data_stats = load_cifar10_stats()
    data_pools = data_stats["pool_3"]
    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)
    
    # Calculate IS
    is_score = tfgan.eval.classifier_score_from_logits(all_logits)
    
    return float(fid), float(is_score)
 
for name in ["evodiff"]: 
# for name in ["dpm_solver++","heun", "uni_pc_bh1", "uni_pc_bh2", "dpm_solver_v3"]:   
    metrics = []
    for step in [5, 6, 8, 10, 12, 15, 20, 25]:  
        path = f"samples/edm-cifar10-32x32-uncond-vp/{name}_{step}"
        fid, is_score = compute_metrics(path)
        metrics.append((step, float(fid), float(is_score)))
    
    # Write results to output file
    with open("output.txt", "a") as f:
        f.write(f"{name} FID: {[m[1] for m in metrics]}\n")
        f.write(f"{name} IS: {[m[2] for m in metrics]}\n")
        # Optionally write in tabular format
        f.write(f"{name} Results:\n")
        f.write("Step\tFID\tIS\n")
        for step, fid, is_score in metrics:
            f.write(f"{step}\t{fid:.4f}\t{is_score:.4f}\n")
        f.write("\n")