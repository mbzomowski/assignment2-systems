import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def distributed_demo(rank, world_size, args):
    setup(rank, world_size)
    d = 1024
    if args.size_tensor is not None:
        size = args.size_tensor
        d = parse_size_string(size)
    d /= 4
    data = torch.randint(0, 1, (d,), dtype=torch.float32, device=f"cuda:{rank}")
    print(f"Rank {rank} data (before all-reduce): {data}")

    dist.all_reduce(data)
    print(f"Rank {rank} data (after all-reduce): {data}")

    cleanup()


def parse_size_string(size_string):
    size_string = size_string.strip().upper()
    if not size_string:
        raise ValueError("Input string cannot be empty.")

    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4,
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
    parser.add_argument("--size_tensor", type=str, help="The size of the torch.Tensors to generate")
    parser.add_argument("--num_tensors", type=int, help="The number of torch.Tensors to generate")

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("Need at least 2 GPUs")
    mp.spawn(distributed_demo, args=(world_size, args,), nprocs=world_size, join=True)
