import torch
from torch.utils.data.dataset import ConcatDataset

import pcode.datasets.mixup_data as mixup
from pcode.datasets.partition_data import DataPartitioner
from pcode.datasets.prepare_data import get_combine_dataset, get_dataset

def load_data_batch(conf, _input, _target, is_training=True):
    
    if conf.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()

    if conf.use_mixup and is_training:
        _input, _target_a, _target_b, mixup_lambda = mixup.mixup_data(
            _input,
            _target,
            alpha=conf.mixup_alpha,
            assist_non_iid=conf.mixup_noniid,
            use_cuda=conf.graph.on_cuda,
        )
        _data_batch = {
            "input": _input,
            "target_a": _target_a,
            "target_b": _target_b,
            "mixup_lambda": mixup_lambda,
        }
    else:
        _data_batch = {"input": _input, "target": _target}
    return _data_batch

def define_dataset(conf, data, display_log=True):
    if conf.prepare_data == "combine":
        combine_dataset = get_combine_dataset(conf, data, conf.data_dir)
        return combine_dataset
    conf.partitioned_by_user = True if "femnist" == conf.data else False
    train_dataset = get_dataset(conf, data, conf.data_dir, split="train")
    test_dataset = get_dataset(conf, data, conf.data_dir, split="test")

    train_dataset, val_dataset, test_dataset = define_val_dataset(
        conf, train_dataset, test_dataset
    )

    if display_log:
        conf.logger.log(
            "Data stat for original dataset: we have {} samples for train, {} samples for val, {} samples for test.".format(
                len(train_dataset),
                len(val_dataset) if val_dataset is not None else 0,
                len(test_dataset),
            )
        )
    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

def define_val_dataset(conf, train_dataset, test_dataset):
    assert conf.val_data_ratio >= 0

    partition_sizes = [
        (1 - conf.val_data_ratio) * conf.train_data_ratio,
        (1 - conf.val_data_ratio) * (1 - conf.train_data_ratio),
        conf.val_data_ratio,
    ]

    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="origin",
        consistent_indices=False,
    )
    train_dataset = data_partitioner.use(0)

    if conf.val_data_ratio > 0:
        assert conf.partitioned_by_user is False

        val_dataset = data_partitioner.use(2)
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, None, test_dataset

def define_data_loader(
    conf, dataset, localdata_id=None, is_train=True, shuffle=True, data_partitioner=None, personal_partition=False, known_distribution=None
):
    if is_train or personal_partition:
        world_size = conf.n_clients
        partition_sizes = [1.0 / world_size for _ in range(world_size)]
        assert localdata_id is not None

        if conf.partitioned_by_user:
            dataset.set_user(localdata_id)
            data_to_load = dataset
        else:
            if data_partitioner is None:
                if personal_partition:
                    data_partitioner = DataPartitioner(
                        conf, dataset, partition_sizes, partition_type="from_known_distribution", known_distribution=known_distribution
                    )
                else:
                    data_partitioner = DataPartitioner(
                        conf, dataset, partition_sizes, partition_type=conf.partition_data 
                    )
            data_to_load = data_partitioner.use(localdata_id)
        if conf.graph.rank == 0:
            conf.logger.log(
                f"Data partition for {'train' if is_train else 'personal test'} (client_id={localdata_id + 1}): partitioned data and use subdata."
            )
    else:
        if conf.partitioned_by_user:
            dataset.set_user(localdata_id)
            data_to_load = dataset
        else:
            data_to_load = dataset
        conf.logger.log("Data partition for validation/test.")

    data_loader = torch.utils.data.DataLoader(
        data_to_load,
        batch_size=conf.batch_size,
        shuffle=shuffle,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    	multiprocessing_context='fork'
    )

    conf.logger.log(
        "\tData stat for {}: # of samples={} for {}. # of batches={}. The batch size={}".format(
            "train" if is_train else "validation/test",
            len(data_to_load),
            f"client_id={localdata_id + 1}" if localdata_id is not None else "Master",
            len(data_loader),
            conf.batch_size,
        )
    )
    conf.num_batches_per_device_per_epoch = len(data_loader)
    conf.num_whole_batches_per_worker = (
        conf.num_batches_per_device_per_epoch * conf.local_n_epochs
    )
    return data_loader, data_partitioner

def define_local_dataset(conf, client_id, datasets, display_log=True):
    assert conf.val_data_ratio >= 0

    local_dataset = datasets.use(client_id-1)

    partition_sizes = [
        conf.train_data_ratio,
        conf.val_data_ratio,
        conf.test_data_ratio,
    ]

    data_partitioner = DataPartitioner(
        conf,
        local_dataset,
        partition_sizes,
        partition_type="random",
    )

    train_dataset = data_partitioner.use(0)
    val_dataset = data_partitioner.use(1)
    test_dataset = data_partitioner.use(2)

    if display_log and conf.graph.rank==0:
        conf.logger.log(
            "Data stat for local dataset on client {}: we have {} samples for train, {} samples for val, {} samples for test.".format(
                client_id,
                len(train_dataset),
                len(val_dataset) if val_dataset is not None else 0,
                len(test_dataset),
            )
        )
    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

def define_local_data_loader(conf, client_id, data_type, data, shuffle=True):

    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=conf.batch_size,
        shuffle=shuffle,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    	multiprocessing_context='fork'
    )

    conf.logger.log(
        "\tData stat for {}: # of samples={} for client {}. # of batches={}. The batch size={}".format(
            data_type,
            len(data),
            client_id,
            len(data_loader),
            conf.batch_size,
        )
    )
    conf.num_batches_per_device_per_epoch = len(data_loader)
    conf.num_whole_batches_per_worker = (
        conf.num_batches_per_device_per_epoch * conf.local_n_epochs
    )
    return data_loader

def define_combine_dataset(conf, dataset):
        world_size = conf.n_clients
        partition_sizes = [1.0 / world_size for _ in range(world_size)]

        data_partitioner = DataPartitioner(conf, dataset, partition_sizes, partition_type=conf.partition_data)
        
        return data_partitioner
