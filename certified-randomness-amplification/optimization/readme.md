This folder contains code for generating random topologies for the circuits and optimize the contraction order for those topologies.
First, generate topologies by running `generate_new_topologies.py`.
Then, optimize the topologies by first creating the `optimization_results/` folder and then running `optimization.py`.
Finally, we can sort the optimized contraction orders and rebuilt the optimized contraction trees using `rebuild_tree.py`.