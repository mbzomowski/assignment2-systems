# benchmark_naive_ddp.py
# -------------------------------------------------------------
# CSE 599O: Distributed Training Basics
#
# Implement a naive DDP version that reproduces the same model
# state as single-process training.
#
# The TA will test your implementation with the following commands:
#
# 1. To verify that DDP matches baseline (toy model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model toy
# Expected output: "Naive DDP matches baseline!"
#
# 2. To output communication and step time (transformer model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model transformer
# Expected output: communication and step time statistics
#
# -------------------------------------------------------------

# Any necessary imports can be added here.
import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from cse599o_basics.optimizer import AdamW
from cse599o_basics.transformerlm import TransformerLM
from tests.common import ToyModel


# Any necessary helper functions can be defined here.
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# You can change the function and variable names as needed.
def run_naive_ddp_worker(rank, world_size, data, targets, num_steps, result_queue, seed):
    """Run one DDP worker process."""
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(seed)
    model = ToyModel().to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = torch.nn.MSELoss()

    split_size = data.shape[0] // world_size
    split_data = torch.split(data, split_size, dim=0)
    data_slice = split_data[rank].to(device)
    split_targets = torch.split(targets, split_size, dim=0)
    target_slice = split_targets[rank].to(device)

    for _ in range(num_steps):
        predictions = model(data_slice)
        loss = loss_fn(predictions, target_slice)
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= world_size
        optimizer.step()

    if rank == 0:
        # move state dict to CPU before sending through Manager queue
        cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        result_queue.put(cpu_state)

    cleanup()


def run_baseline(data, targets, num_steps, seed):
    """Run single-process baseline for comparison."""
    device = torch.device("cuda:0")

    torch.manual_seed(seed)
    model = ToyModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    loss_fn = torch.nn.MSELoss()

    data = data.to(device)
    targets = targets.to(device)

    for _ in range(num_steps):
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # return CPU copy of the state dict for easy comparison
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# You can change the function and variable names as needed.
def verify_naive_ddp():
    """Benchmark and verify naive DDP."""
    world_size = 2
    num_steps = 5
    data = torch.randn(10, 10)
    targets = torch.randn(10, 5)
    seed = 1234

    # Run baseline
    no_ddp_state = run_baseline(data, targets, num_steps, seed)

    # Set up multiprocessing for DDP
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_naive_ddp_worker,
        args=(world_size, data, targets, num_steps, result_queue, seed),
        nprocs=world_size,
        join=True,
    )

    # Get model state from DDP
    ddp_state = result_queue.get()

    assert len(no_ddp_state) > 0, "model state from baseline is empty"
    for name in no_ddp_state:
        assert torch.allclose(no_ddp_state[name], ddp_state[name], atol=1e-6)
    print("Naive DDP matches baseline!")


def run_transformer_ddp(rank, world_size, data, targets, num_steps, result_queue):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    model = TransformerLM(1000, 10, 1280, 36, 20, 5120, 1e4).to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = torch.nn.MSELoss()

    split_size = data.shape[0] // world_size
    split_data = torch.split(data, split_size, dim=0)
    data_slice = split_data[rank].to(device)
    split_targets = torch.split(targets, split_size, dim=0)
    target_slice = split_targets[rank].to(device)

    dist.barrier()

    for _ in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.time()
        gradient_time = 0
        predictions = model(data_slice)
        loss = loss_fn(predictions, target_slice)
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                torch.cuda.synchronize()
                t1 = time.time()
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()
                t2 = time.time()
                p.grad.data /= world_size
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.time()
        total_time = t3 - t0
        time_frac = (t2 - t1) / total_time

        if rank == 0:
            print(f"\nIteration {_}\nTotal training time: {total_time:.2}\nFraction of time spent on all_reduce: {time_frac:.2}")

    cleanup()


# You can change the function and variable names as needed.
def timing_naive_ddp():
    """Timing benchmark for naive DDP with transformer model."""
    world_size = 2
    num_steps = 5
    data = torch.randint(0, 1000, (10, 10))
    targets = torch.randint(0, 1000, (10, 10))

    # Set up multiprocessing for DDP
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_transformer_ddp,
        args=(world_size, data, targets, num_steps, result_queue),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["toy", "transformer"], default="toy")
    args = parser.parse_args()

    if args.model == "toy":
        verify_naive_ddp()
    elif args.model == "transformer":
        timing_naive_ddp()
