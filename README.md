# SPAlign
This repository contains the official implementation of the paper:

**SPAlign—Structure-Aware Smooth Perception for Heterogeneous Feature Alignment**

## Requirements

Please run the following commands below to install dependencies.

```bash
conda create -y -n spalin python=3.7
conda activate spalin
conda install -y -c pytorch pytorch=1.7.1 torchvision=0.8.2 matplotlib python-lmdb cudatoolkit=11.3 cudnn
pip install transformers datasets pytreebank opencv-python torchcontrib gpytorch 
```

## Training

```bash
python run_gloo.py --arch simple_cnn --complex_arch master=simple_cnn,worker=simple_cnn --experiment demo --data cifar100 --pin_memory True --batch_size 64 --num_workers 2 --prepare_data combine --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --train_data_ratio 0.8 --val_data_ratio 0.0 --test_data_ratio 0.2 --n_clients 20 --participation_ratio 1 --n_comm_rounds 50 --local_n_epochs 5 --world_conf 0,0,1,1,100 --on_cuda True --fl_aggregate scheme=federated_average --optimizer sgd --lr 0.01 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 --lr_scheduler MultiStepLR --lr_decay 0.1 --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 --track_time True --display_tracked_time True --python_path /home/miniconda3/envs/spalin/bin/python --manual_seed 7 --pn_normalize True --same_seed_process True --algo SPAlign --personal_test True --w_conv_bias 0.0 --w_fc_bias 0.0 --data_dir ./data --use_fake_centering False --port 20002 --timestamp $(date '+%Y%m%d%H%M%S') --lamda 0.5 --self_distillation_temperature 6.0 --alpha 0.6 > logs/log.txt 2>&1
```

