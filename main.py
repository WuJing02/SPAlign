import numpy as np

import torch
import torch.distributed as dist

from parameters import get_args
from pcode.master import Master
from pcode.worker import Worker
import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser

def main(conf):
    try:
        dist.init_process_group("mpi")
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    init_config(conf)

    process = Master(conf) if conf.graph.rank == 0 else Worker(conf)
    print("rank is "+conf.graph.rank)
    process.run()

def init_config(conf):
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = dist.get_rank()

    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)

    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(conf.manual_seed)
        torch.cuda.set_device(conf.graph.primary_device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True if conf.train_fast else False
        torch.cuda.manual_seed_all(conf.manual_seed)

    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))

    conf.logger = logging.Logger(conf.checkpoint_dir)

    if conf.graph.rank == 0:
        logging.display_args(conf)

    dist.barrier()

if __name__ == "__main__":
    conf = get_args()
    main(conf)