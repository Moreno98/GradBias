# MULTI DATA PARALLELIZATION
from torch.distributed import init_process_group, destroy_process_group
import os
import datetime

class DDP():
    def __init__(
        self,
        rank,
        world_size
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = f"cuda:{rank}"
        self.ddp_setup(rank, world_size)
        self.main()
        destroy_process_group()
    
    def main(self):
        pass

    def ddp_setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12354"
        init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))

class DDP_Node():
    def __init__(
        self,
        ranks,
        world_size,
        world_size_node
    ):
        self.ranks = ranks
        self.world_size = world_size_node
        self.devices = [f"cuda:{rank}" for rank in ranks]
        self.ddp_setup(ranks[0], world_size)
        self.main()
        destroy_process_group()
    
    def main(self):
        pass

    def ddp_setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12354"
        init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=5400))



