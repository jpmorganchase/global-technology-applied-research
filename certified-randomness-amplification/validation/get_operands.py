###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import json
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('../')
from src.circuit import get_circuit


# Utility for converting integer storage format of measurement result to bitstring
def int_str_to_bin(n, bitstring):
    result = bin(int(bitstring))[2:]
    result = '0' * (n - len(result)) + result
    return result


if __name__ == "__main__":
    # Generating operands

    n = 64
    get_operands = True
    get_tree = True

    with open(f'contraction_scheme.json', 'r') as file:
        scheme = json.load(file)

    topology = scheme['topology']

    with open("experiment_output.txt", "r") as file:
        lines = file.readlines()
        
    verify = np.load('initial_verification_set.npy')
    verify2 = np.load('second_verification_set.npy')

    if get_operands:
        pass
    else:
        lines = lines[:1]

    i = 0
    for j, line in tqdm(enumerate(lines)):
        try:
            line = line.lstrip()
            if len(line) == 0:
                continue
            circuit_stream = line.split(" ")
            assert len(circuit_stream) == 13
            assert circuit_stream[0].split(":")[0] == f'R{j}'
            if get_operands and verify[j]:
                leakage = circuit_stream[-1].split(":")[1]
                gate_layers = circuit_stream[:-3]
                bitstring = circuit_stream[-2].split(":")[1][1:]
                meas = circuit_stream[-3].split(":")[1][1:]
                sq_gates_list = [gate_layer.split(":")[2] for gate_layer in gate_layers]
                quimb_circ = get_circuit(n, topology, sq_gates_list)
                for q, m in enumerate(meas):
                    if m=="1":
                        quimb_circ.apply_gate('H', q)
                rehs = quimb_circ.amplitude_rehearse(b=int_str_to_bin(n, bitstring), optimize='greedy')
                tensors = rehs['tn']
                tensors = [tensor.data for tensor in tensors.tensors]
                np.save(f'operands/operands_{i}.npy', np.array(tensors))
                i += 1
        except:
            print(f'Round {j} failed, {verify[j]}')
    
    for j, line in tqdm(enumerate(lines)):
        try:
            line = line.lstrip()
            if len(line) == 0:
                continue
            circuit_stream = line.split(" ")
            assert len(circuit_stream) == 13
            assert circuit_stream[0].split(":")[0] == f'R{j}'
            if get_operands and verify2[j]:
                leakage = circuit_stream[-1].split(":")[1]
                gate_layers = circuit_stream[:-3]
                bitstring = circuit_stream[-2].split(":")[1][1:]
                meas = circuit_stream[-3].split(":")[1][1:]
                sq_gates_list = [gate_layer.split(":")[2] for gate_layer in gate_layers]
                quimb_circ = get_circuit(n, topology, sq_gates_list)
                for q, m in enumerate(meas):
                    if m=="1":
                        quimb_circ.apply_gate('H', q)
                rehs = quimb_circ.amplitude_rehearse(b=int_str_to_bin(n, bitstring), optimize='greedy')
                tensors = rehs['tn']
                tensors = [tensor.data for tensor in tensors.tensors]
                np.save(f'operands/operands_{i}.npy', np.array(tensors))
                i += 1
        except:
            print(f'Round {j} failed, {verify[j]}')


    if get_tree:
        import cotengra
        import pickle

        def slice_provided_indices(tree: cotengra.ContractionTree, ix_sl):
                for ix in ix_sl:
                    tree.remove_ind_(ix)
                return tree

        def reload_contraction_scheme(tree, path, sliced_inds):
            inputs, output, size_dict = tree.inputs, tree.output, tree.size_dict
            tree = slice_provided_indices(cotengra.ContractionTree.from_path(inputs, output, size_dict, path=path), sliced_inds)
            return tree

        path = scheme['path']
        sliced_inds = scheme['sliced_inds']
        tree = rehs['tree']
        tree = reload_contraction_scheme(tree, path, sliced_inds)

        with open('tree.pkl', 'wb') as f:
            pickle.dump(tree, f)