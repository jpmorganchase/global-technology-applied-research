###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import cotengra
import pickle
from tqdm import tqdm
import concurrent
import time
import numpy as np
import sys
sys.path.append('../')
from ..src.circuit import get_circuit

workers = 128


def to_tuple(my_list):
    if type(my_list[0]) == list:
        output = []
        for layer in my_list:
            output.append(to_tuple(layer))
        return tuple(output)
    else:
        return tuple(my_list)


print('Creating topology dictionary')
with open(f'topologies.pkl', 'rb') as f:
    topology_list = pickle.load(f)


n = 64
tree_dict = {}
def build_tree_task(topology):
    depth = len(topology)
    sq_gates_list = np.random.randint(0, 8, (depth + 1, n))
    circuit = get_circuit(n, topology, sq_gates_list)
    rehs = circuit.amplitude_rehearse(optimize='greedy')
    tree = rehs['tree']
    return topology, tree


with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(build_tree_task, topology) for topology in topology_list}
    for future in tqdm(concurrent.futures.as_completed(futures)):
        topology, tree = future.result()
        tree_dict[to_tuple(topology[2])] = tree


def slice_provided_indices(tree: cotengra.ContractionTree, ix_sl):
    for ix in ix_sl:
        tree.remove_ind_(ix)
    return tree


def recover_tree(topology, path, slices):
    tree = tree_dict[to_tuple(topology[2])]
    inputs, output, size_dict = tree.inputs, tree.output, tree.size_dict
    tree = slice_provided_indices(cotengra.ContractionTree.from_path(inputs, output, size_dict, path=path), slices)
    return tree, topology, path, slices


print('Loading optimization results')
with open('optimization_results/results.pkl', 'rb') as f:
    results = pickle.load(f)

print('Total results: ', len(results))

def task(result):
    topology, path, slices = result
    return recover_tree(topology, path, slices)


rebuilt_trees = []
with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
    start = time.time()
    futures = {executor.submit(task, result) for result in results}
    prev_time = time.time()
    print(f'Submit time {prev_time - start}')
    i = 0
    print(executor._max_workers)
    for future in tqdm(concurrent.futures.as_completed(futures)):
        i += 1
        rebuilt_trees.append(future.result())



flops = []
costs = []
for tree, topology, path, slices in tqdm(rebuilt_trees):
    flops.append([tree.total_flops('complex64'), topology, path, slices])
    costs.append([tree.total_cost(1024), topology, path, slices])


with open(f'optimization_results/flops_and_tree_info.pkl', 'wb') as f:
    pickle.dump(flops, f)


with open(f'optimization_results/costs_and_tree_info.pkl', 'wb') as f:
    pickle.dump(costs, f)