
import os
import subprocess
import time
import torch


lamda = 0.5
kl_t = 6.0
alpha = 0.6

base_cmd = (
    "python run_gloo.py "
    "--arch simple_cnn "
    "--complex_arch master=simple_cnn,worker=simple_cnn "
    "--experiment demo "
    "--data cifar100 "
    "--pin_memory True "
    "--batch_size 64 "
    "--num_workers 2 "
    "--prepare_data combine "
    "--partition_data non_iid_dirichlet "
    "--non_iid_alpha 0.01 "
    "--train_data_ratio 0.8 "
    "--test_data_ratio 0.2 "
    "--n_clients 20 "
    "--participation_ratio 0.6 "
    "--n_comm_rounds 50 "
    "--local_n_epochs 5 "
    "--world_conf 0,0,1,1,100 "
    "--on_cuda True "
    "--fl_aggregate scheme=federated_average "
    "--optimizer sgd "
    "--lr 0.01 "
    "--lr_warmup False "
    "--lr_warmup_epochs 5 "
    "--lr_warmup_epochs_upper_bound 150 "
    "--lr_scheduler MultiStepLR "
    "--lr_decay 0.1 "
    "--weight_decay 1e-5 "
    "--use_nesterov False "
    "--momentum_factor 0.9 "
    "--track_time True "
    "--display_tracked_time True "
    "--python_path /home/miniconda3/envs/spalign/bin/python "
    "--manual_seed 7 "
    "--pn_normalize True "
    "--same_seed_process True "
    "--algo SPAlign "
    "--personal_test True "
    "--data_dir ./data "
    "--use_fake_centering False "
)

port = 20002
log_file = "logs/log.txt"

os.makedirs("logs", exist_ok=True)

if os.path.exists(log_file):
    print(f"Deleting previous log file: {log_file}")
    os.remove(log_file)

cmd = (
    f"{base_cmd} "
    f"--port {port} "
    f"--timestamp $(date '+%Y%m%d%H%M%S') "
    f"--lamda {lamda} "
    f"--self_distillation_temperature {kl_t} "
    f"--alpha {alpha} "
    f"> {log_file} 2>&1"
)

print(f"Running combination: lamda={lamda}, kl_t={kl_t}, alpha={alpha}")
print(f"Command: {cmd}")

proc = subprocess.Popen(cmd, shell=True)
returncode = proc.wait()

if returncode != 0:
    print(f"Warning: Process exited with non-zero status ({returncode}). Check {log_file} for details.")

try:
    torch.cuda.empty_cache()
except Exception as e:
    print(f"Warning: Failed to clear GPU memory: {e}")

print("Waiting for GPU and system resources to stabilize...")
time.sleep(15)


print("Experiment completed.")
