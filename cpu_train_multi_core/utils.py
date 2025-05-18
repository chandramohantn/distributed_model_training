import os
import torch.distributed as dist


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size
    )

def cleanup_ddp():
    dist.destroy_process_group()
