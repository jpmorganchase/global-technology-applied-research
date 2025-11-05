###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import sys
sys.path.append('../')
from src.circuit import get_random_topology
from tqdm import tqdm
import pickle
import concurrent
import numpy as np

workers = 128


def task(seed):
    np.random.seed(seed)
    topology = get_random_topology(64, 9, True, 20, True, 10000)
    return topology


graph_topologies = []
topologies = []
with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(task, seed) for seed in np.random.randint(0, 2**32, size=128)}
    for future in tqdm(concurrent.futures.as_completed(futures)):
        topology = future.result()
        topologies.append(topology)


with open('topologies.pkl', 'wb') as f:
    pickle.dump(topologies, f)