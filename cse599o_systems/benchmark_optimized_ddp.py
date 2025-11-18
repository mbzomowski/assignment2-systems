# benchmark_optimized_ddp.py
# -------------------------------------------------------------
# CSE 599O
#
# Extend your DDP benchmark to evaluate three optimized variants
# for the Transformer model:
#   (1) run_flat
#   (2) run_individual
#   (3) run_bucketed
#
# The TA will execute your script using commands like:
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode flat
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode individual
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode bucketed --bucket-mb 10
#
# Each function should measure and print out the following statistics:
#   - iteration time per step  → append to iteration_times
#   - communication time per step → append to comm_times
# -------------------------------------------------------------

import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cse599o_basics.optimizer import AdamW
from cse599o_basics.transformerlm import TransformerLM
# Any other necessary imports can be added here.

# Any necessary helper functions can be defined here.
def run_flat_worker(rank, model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    inputs, targets = data
    inputs = inputs.to(device)
    targets = targets.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    split_size = inputs.shape[0] // world_size
    split_data = torch.split(inputs, split_size, dim=0)
    input_slice = split_data[rank].to(device)
    split_targets = torch.split(targets, split_size, dim=0)
    target_slice = split_targets[rank].to(device)

    dist.barrier()

    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        optimizer.zero_grad()

        logits = model(input_slice)
        B, S, V = logits.shape
        logits_flat = logits.view(B * S, V)
        targets_flat = target_slice.view(B * S)

        loss = loss_fn(logits_flat, targets_flat)
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.time()

        grads = []
        shapes = []
        sizes = []
        for p in model.parameters():
            if p.grad is not None:
                shapes.append(p.grad.data.size())
                sizes.append(p.grad.data.numel())
                grads.append(p.grad.data.view(-1))
            else:
                shapes.append(0)
                sizes.append(0)
                grads.append(torch.empty())
        flattened_tensors = torch.cat(grads, dim=0).to(device)
        dist.all_reduce(flattened_tensors, op=dist.ReduceOp.SUM)
        flattened_tensors /= world_size

        for p, sh, sz in zip(model.parameters(), shapes, sizes):
            if p.grad is not None:
                t = flattened_tensors[:sz].view(sh)
                p.grad.data = t.detatch().clone()
                flattened_tensors = flattened_tensors[sz:]
            else:
                assert sz == 0
                assert sh == 0

        torch.cuda.synchronize()
        t2 = time.time()

        optimizer.step()

        torch.cuda.synchronize()
        t3 = time.time()
        total_time = t3 - t0
        time_frac = (t2 - t1) / total_time

        if rank == 0:
            print(f"\nIteration {_}\nTotal training time: {total_time:.5}\nFraction of time spent on all_reduce: {time_frac:.5}")

        cleanup()


def cleanup():
    dist.destroy_process_group()


# You can change the function and variable names as needed.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# ============================================================
# (0) Naive DDP
# ============================================================
# You can change the function and variable names as needed.
def run_naive(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """A naive DDP training loop for reference."""
    # TODO:
    pass


# ============================================================
# (1) Flat DDP
# ============================================================
# You can change the function and variable names as needed.
def run_flat(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """All-reduce a single flattened gradient tensor."""
    world_size = 2
    mp.spawn(
        run_flat_worker,
        args=(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times, world_size),
        nprocs=world_size,
        join=True,
    )


# ============================================================
# (2) Individual DDP
# ============================================================
# You can change the function and variable names as needed.
def run_individual(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times):
    """All-reduce each parameter's gradient individually."""
    # TODO:
    pass


# ============================================================
# (3) Bucketed DDP
# ============================================================
# You can change the function and variable names as needed.
def run_bucketed(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times, bucket_mb):
    """Group gradients into buckets and all-reduce each bucket."""
    # TODO:
    pass


# ============================================================
# Benchmark Function
# ============================================================
# You can change the function and variable names as needed.
def benchmark_optimized_ddp():
    """Benchmark DDP variants on the Transformer model."""
    parser = argparse.ArgumentParser(description="Benchmark optimized DDP variants.")
    parser.add_argument(
        "--mode",
        type=str,
        default="flat",
        choices=["flat", "individual", "bucketed"],
        help="Select which DDP variant to benchmark.",
    )
    parser.add_argument(
        "--bucket-mb",
        type=int,
        default=10,
        help="Bucket size (in MB) for the bucketed DDP variant.",
    )
    args = parser.parse_args()

    # Example placeholders
    num_iters, num_warmup = 5, 2
    iteration_times, comm_times = [], []
    # Model parameters
    d_model = 1280
    d_ff = 5120
    num_layers = 36
    num_heads = 20
    # Optimizer parameters
    lr = 1e-3
    weight_decay = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8

    # DDP setup
    mp.set_start_method("spawn", force=True)

    # Construct model and move to GPU
    # TODO: Define model parameters
    model = TransformerLM(1000, 10, d_model, num_layers, num_heads, d_ff, 1e4)

    # Construct optimizer
    # TODO: Define optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    # Dummy data
    # TODO: Create input data
    inputs = torch.randint(0, 1000, (10, 10))
    targets = torch.randint(0, 1000, (10, 10))
    data = (inputs, targets)

    if args.mode == "naive":
        run_naive(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "flat":
        run_flat(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "individual":
        run_individual(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times)
    elif args.mode == "bucketed":
        run_bucketed(model, data, optimizer, num_iters, num_warmup, iteration_times, comm_times, args.bucket_mb)

    print(f"Mode: {args.mode}")
    print(f"Iteration times: {iteration_times}")
    print(f"Communication times: {comm_times}")


if __name__ == "__main__":
    benchmark_optimized_ddp()
