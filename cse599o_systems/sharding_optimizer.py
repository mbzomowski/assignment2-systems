# sharding_optimizer.py
# -------------------------------------------------------------
# CSE 599O: 
#
# Implement optimizer state sharding for distributed training.
#
# -------------------------------------------------------------
import os
import torch
import torch.distributed as dist
import argparse
import torch.multiprocessing as mp
from multiprocessing import Manager
from timeit import default_timer as timer
# You can add other necessary imports here.


class SharedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        # TODO: Initialize the underlying optimizer
    
    def step(self, closure=None, **kwargs):
       # TODO: Implement optimizer step with sharded states

    # You can add other necessary methods here.


# Add any necessary helper functions here.

# You can change the function and variable names as needed.
def run_distributed_training(rank, world_size, num_trials, num_warmup_trials, result_queue):
    # Setup distributed environment
    # TODO

    # Construct model
    # TODO

    # Construct random input data
    # TODO: Create input data

    # Construct optimizer
    # You can use the SharedOptimizer here
    # TODO
    
    # Training loop
        # Warm up
        # TODO
        # Benchmark
        # TODO
    
    if rank == 0:
       # Collect and return the timing results

if __name__ == "__main__":
    # Set up distributed training parameters
    # Collect results and print timing summary
    # TODO