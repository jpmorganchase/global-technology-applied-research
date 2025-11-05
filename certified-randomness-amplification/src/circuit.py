###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import quimb.tensor as qtn
from math import pi
import numpy as np

my_pi = pi

# Create a quimb circuit from topology and single-qubit gate list
def get_circuit(n, topology, sq_gates_list) -> qtn.Circuit:
    # Definition of PhasedX gate according to guppy
    # $$  \mathrm{Rz(\theta_2)Rx(\theta_1)Rz(-\theta_2)} =
    # \begin{pmatrix}
    # \cos(\frac{ \theta_1}{2}) &
    # -i e^{-i \theta_2}\sin(\frac{\theta_1}{2})\\
    # -i e^{i \theta_2}\sin(\frac{\theta_1}{2}) &
    # \cos(\frac{\theta_1}{2})
    # \end{pmatrix} $$
    # Cotengra U3 gate convention https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.U3Gate
    # lam = np.pi/2 - theta2
    # phi = - lam
    assert n % 2 == 0
    circuit = qtn.Circuit(n)
    def rand_sq_layer(sq_gates):
        for i, gate_index in enumerate(sq_gates):
            theta1 = my_pi / 2
            theta2 = my_pi * (int(gate_index)-4)/4
            theta = theta1
            phi = - pi / 2 + theta2
            lam =  - phi
            circuit.apply_gate('U3', theta, phi, lam, i, parameterize=True)
    def rand_tq_layer(two_qubit_gate_pairs):
        for qubits_for_gate in two_qubit_gate_pairs:
            circuit.apply_gate('RZZ', my_pi/2, qubits_for_gate[0], qubits_for_gate[1])

    for l in range(len(topology)):
        sq_gates = sq_gates_list[l]
        two_qubit_gate_pairs = topology[l]
        rand_sq_layer(sq_gates)
        rand_tq_layer(two_qubit_gate_pairs)
    rand_sq_layer(sq_gates_list[-1])
    return circuit


# Count the light cone of a qubit for a circuit
def future_light_cone(topology_or_circuit, input_qubit):
    affected_qubits = {input_qubit}
    if type(topology_or_circuit) == qtn.Circuit:
        circuit = topology_or_circuit
        for gate in circuit.gates:
            if any(q in affected_qubits for q in gate.qubits):
                affected_qubits.update(gate.qubits)
    elif type(topology_or_circuit) == list:
        topology = topology_or_circuit
        for layer in topology:
            for gate in layer:
                if any(q in affected_qubits for q in gate):
                    affected_qubits.update(gate)
    return affected_qubits


# Generate a random topology, potentially subject to the requirement that all qubits have full light cone coverage
def get_random_topology(n, depth, final_gates=None, light_cone_coverage=False, max_counter=100):
    if light_cone_coverage == False:
        topology = []
        for _ in range(depth):
            qubits = [int(j) for j in np.random.permutation([i for i in range(n)])]
            topology.append(list(zip(qubits[::2], qubits[1::2])))
        if final_gates is not None:
            for _ in range(n // 2 - final_gates):
                topology[-1].pop()
        return topology
    else:
        future_light_cone_size = 0
        counter = 0
        while future_light_cone_size != n * n:
            if counter == max_counter:
                raise RuntimeError(f"topology generation failed after {max_counter} iterations")
            counter += 1
            topology = get_random_topology(n, depth, final_gates=final_gates)
            future_light_cone_size = 0
            for i in range(n):
                future_light_cone_size += len(future_light_cone(topology, i))
        return topology
    

# Get random quimb circuit, potentially subject to the requirement that all qubits have full light cone coverage
def get_random_circuit(n, depth, final_gates=None, light_cone_coverage=False, max_counter=100):
    '''
    Parameters
    ----------
    n : int
        The number of qubits
    depth : int
        The number of two-qubit gate layers, which is the two-qubit gate depth.
    final_gates : int, optional
        The number of two-qubit gates in the final layer. The default is n/2.
    light_cone_coverage : bool, optional
        Whether every qubit is required to have its light cone covering all other qubits.
    max_counter : int, optional
        The maximum number of iterations allowed to retry to satisfy the light cone requirement, if it is True.
    '''
    topology = get_random_topology(n, depth, final_gates=final_gates, light_cone_coverage=light_cone_coverage, max_counter=max_counter)
    sq_gates_list = np.random.randint(0, 8, (depth + 1, n))
    circuit = get_circuit(n, topology, sq_gates_list)
    return circuit