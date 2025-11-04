import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, worldsize):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = 29500
    dist.init_process_group(backend='nccl', rank=rank, worldsize=worldsize)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def distributed_demo(rank, worldsize):
    setup(rank, worldsize)
    data = torch.randint(0, 10, (3,), device=f"cuda:{rank}")
    print(f"Rank {rank} data (after all-reduce): {data}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("Need at least 2 GPUs")
    mp.spawn(distributed_demo, args=(world_size,), nprocs=world_size, join=True)
