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
from cse599o_basics.optimizer import AdamW
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
    optimizer = AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()

    split_size = data.shape[0] // world_size
    split_data = torch.split(data, split_size, dim=0)
    data_slice = split_data[rank].to(device)
    split_targets = torch.split(targets, split_size, dim=0)
    target_slice = split_targets[rank].to(device)

    for _ in num_steps:
        predictions = model(data_slice)
        loss = loss_fn(predictions, target_slice)
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            dist.all_reduce(p, op=dist.ReduceOp.MEAN)
        optimizer.step()

    if rank == 0:
        # TODO: Collect and return the model state from rank 0
        result_queue.put({model.state_dict()})  # Replace with actual model state

    cleanup()


def run_single_worker(rank, world_size, data, targets, num_steps, result_queue, seed):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(seed)
    model = ToyModel().to(device)
    optimizer = AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()

    for _ in num_steps:
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    result_queue.put({model.state_dict()})

    cleanup()


# You can change the function and variable names as needed.
def run_baseline(data, targets, num_steps, seed):
    """Run single-process baseline for comparison."""
    manager = mp.Manager()
    result_queue = manager.Queue()
    mp.spawn(
        run_single_worker,
        args=(1, data, targets, num_steps, result_queue),
        nprocs=1,
        join=True,
    )
    return {result_queue.get()}  # Replace with actual model state


# You can change the function and variable names as needed.
def verify_naive_ddp():
    """Benchmark and verify naive DDP."""
    world_size = 2
    num_steps = 5
    data = torch.randn(10, 5)
    targets = torch.randn(5)
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


# You can change the function and variable names as needed.
def timing_naive_ddp():
    """Timing benchmark for naive DDP with transformer model."""
    # TODO
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["toy", "transformer"], default="toy")
    args = parser.parse_args()

    if args.model == "toy":
        verify_naive_ddp()
    elif args.model == "transformer":
        timing_naive_ddp()
