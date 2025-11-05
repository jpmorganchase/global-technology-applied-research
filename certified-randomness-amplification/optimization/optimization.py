###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import cotengra
import pickle
import concurrent
import time
import sys
sys.path.append('../')
from ..src.circuit import get_circuit





if __name__ == "__main__":

    workers = 128

    with open('topologies.pkl', 'rb') as f:
        topology_list = pickle.load(f)


    topology_list = [topology_list[1]]

    hyper_params = {
        "methods": ['kahypar', 'greedy'],
        "max_repeats": 100,
        "max_time": 10000,
        "minimize": 'combo-1024',
        "parallel": False,
        "progbar": False,
        "slicing_reconf_opts": {
            "target_size": 2**28,
        },
        "optlib": "random",
    }


    n = 64

    def trial(topology):
        depth = len(topology)
        sq_gates_list = np.random.randint(0, 8, (depth + 1, n))
        circuit = get_circuit(n, topology, sq_gates_list)
        rehs = circuit.amplitude_rehearse(optimize='greedy')
        tree = rehs['tree']
        eq = tree.get_eq()
        shapes = [(2, 2) for _ in eq.split(',')]
        opt = cotengra.HyperOptimizer(**hyper_params)
        tree = cotengra.einsum_tree(eq, *shapes, optimize=opt)
        path, sliced_inds = tree.get_path(), tree.sliced_inds
        slices = []
        for ind in sliced_inds:
            slices.append(ind)
        # Avoid directly saving the contraction tree since it is very heavy
        return topology, path, slices

    num_finished = 0
    results = []
    try:
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print('No prior results exist')
    last_save_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        print(executor._max_workers)
        futures = {executor.submit(trial, topology) for topology in topology_list * 10000}
        for future in concurrent.futures.as_completed(futures):
            num_finished += 1
            results.append(future.result())
            if time.time() - last_save_time > 10:
                last_save_time = time.time()
                with open('optimization_results/backup.pkl', 'wb') as f:
                    pickle.dump(results, f)
                with open('optimization_results/results.pkl', 'wb') as f:
                    pickle.dump(results, f)
                print(f'Finished {num_finished} tasks.')
                sys.stdout.flush()