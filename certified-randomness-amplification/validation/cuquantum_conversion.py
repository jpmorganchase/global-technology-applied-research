###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from cuquantum import Network

def get_path_and_slices(tree):
    path = tree.get_path()
    sliced_inds = list(tree.sliced_inds)
    slices = []
    for ind in sliced_inds:
        slices.append((ind, 1))
    slices = tuple(slices)
    return path, slices


def cuquantum_network_from_cotengra_tree(tree, operands, device_id):
    path, slices = get_path_and_slices(tree)
    eq = tree.get_eq()
    network = Network(eq, *operands, options={'device_id' : device_id})
    path, info = network.contract_path(optimize={'path': path, 'slicing': slices})
    return network, info