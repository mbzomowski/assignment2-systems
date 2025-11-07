import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import time


WARMUP_ROUNDS = 5


def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def distributed_run(rank, world_size, args):
    setup(rank, world_size, args)
    size = args.tensor_size
    d = int(parse_size_string(size) / 4)
    data = torch.randn(d, dtype=torch.float32, device=f"cuda:{rank}")
    for r in range(WARMUP_ROUNDS):
        print(f"**********  WARMUP  **********  ROUND {r}")
        dist.all_reduce(data)

    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.time()
    dist.all_reduce(data)
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = t1 - t0
    t = torch.tensor([elapsed], device=f"cuda:{rank}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"world_size={world_size}, size={size}, "
            f"avg_all_reduce_max_over_ranks={t.item():.6f}s"
        )

    cleanup()


def parse_size_string(size_string):
    size_string = size_string.strip().upper()
    if not size_string:
        raise ValueError("Input string cannot be empty.")

    multipliers = {
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
    }

    for unit, multiplier in multipliers.items():
        if size_string.endswith(unit):
            value_str = size_string[:-len(unit)].strip()
            try:
                value = float(value_str)
                return int(value * multiplier)
            except ValueError:
                raise ValueError(f"Invalid numeric value in size string: {value_str}")

    raise ValueError(f"Unknown size unit in string: {size_string}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that runs distributed communication on a single node")
    parser.add_argument("--tensor_size", type=str, default='1KB', help="The size of the torch.Tensors to generate")
    parser.add_argument("--num_tensors", type=int, default=2, help="The number of torch.Tensors to generate")
    parser.add_argument("--backend", type=str, default='nccl', help="nccl or gloo")

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("Need at least 2 GPUs")
    mp.spawn(distributed_run, args=(world_size, args,), nprocs=world_size, join=True)
