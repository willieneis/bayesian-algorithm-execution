"""
Testing out: active ranking via pairwise comparisons (using BAX with sorting
algorithms).
"""
import numpy as np
import matplotlib.pyplot as plt

# Set seed
seed = 11
np.random.seed(seed)

# Set up problem
n_items = 11

# Generate embeddings
bounds = [0, 10]
item_embed = list(np.random.uniform(bounds[0], bounds[1], n_items))

# Plot items
fig, ax = plt.subplots(figsize=(5, 5))
for item_1 in item_embed:
    for item_2 in item_embed:
        color = 'b' if item_1 > item_2 and item_1 < bounds[1] - item_2 else 'r'
        ax.plot([item_1], [item_2], 'o', color=color)

ax.set(ylim=(-1, 11), xlim=(-1, 11))
plt.show()
