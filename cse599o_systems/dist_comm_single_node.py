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


def warmup(rank, world_size, args):
    setup(rank, world_size, args)
    size = args.tensor_size
    d = int(parse_size_string(size) / 4)

    print("**********  WARMUP  **********")
    for r in range(WARMUP_ROUNDS):
        data = torch.randn(d, dtype=torch.float32, device=f"cuda:{rank}")
        print(f"Round {r}: Rank {rank} data (before all-reduce): {data}")
        dist.all_reduce(data)
        print(f"Round {r}: Rank {rank} data (after all-reduce): {data}")
    cleanup()


def distributed_run(rank, world_size, args):
    setup(rank, world_size, args)
    size = args.tensor_size
    d = int(parse_size_string(size) / 4)
    data = torch.randn(d, dtype=torch.float32, device=f"cuda:{rank}")
    print(f"Rank {rank} data (before all-reduce): {data}")

    dist.all_reduce(data)

    print(f"Rank {rank} data (after all-reduce): {data}")
    print(f"Time: {end_time-start_time}")

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
    mp.spawn(warmup, args=(world_size, args,), nprocs=world_size, join=True)
    torch.cuda.synchronize()
    start_time = time.time()
    mp.spawn(distributed_run, args=(world_size, args,), nprocs=world_size, join=True)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Total time taken: {end_time-start_time}")
