import matplotlib.pyplot as plt
from collections import defaultdict


a = (2, 1, 0.005081)
b = (4, 1, 0.014847)
c = (8, 1, 0.036020)

d = (2, 10, 0.007829)
e = (4, 10, 0.014484)
f = (8, 10, 0.031778)

g = (2, 100, 0.014668)
h = (4, 100, 0.030914)
i = (8, 100, 0.058664)

j = (2, 1000, 0.101967)
k = (4, 1000, 0.164307)
l = (8, 1000, 0.241136)

# Collect all points
points = [a, b, c, d, e, f, g, h, i, j, k, l]

# Group by the "MB" size (second element)

grouped = defaultdict(list)
for gpus, mb, secs in points:
    grouped[gpus].append((mb, secs))

# Plot one graph per MB size
for gpus in sorted(grouped.keys()):
    # Sort by GPU count for nicer lines
    grouped[gpus].sort(key=lambda x: x[0])
    mbs = [x for x, _ in grouped[gpus]]
    secs = [y for _, y in grouped[gpus]]

    plt.figure()
    plt.xscale('log', base=2)
    plt.plot(mbs, secs, marker='o')
    plt.xlabel("MB")
    plt.ylabel("seconds")
    plt.title(f"{gpus} GPUs")
    plt.grid(True)

plt.show()
