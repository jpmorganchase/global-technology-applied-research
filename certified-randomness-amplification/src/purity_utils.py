###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from pytket import Circuit, OpType

import numpy as np
import random
import stim
import kahypar


def partition(n, d, k, pairs):
    
    edges_sep = pairs
    edges = [edges_sep[i][j] for i in range(d) for j in range(n//2)]

    num_nodes = n
    num_nets = d*n//2

    hyperedges = list(np.reshape(np.array(edges),(n*d)))
    hyperedges = [int(item) for item in hyperedges]
    hyperedge_indices = list(range(0,len(hyperedges)+1,2))

    node_weights = [1 for j in range(num_nodes)]
    edge_weights = [1 for j in range(num_nets)]

    hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

    context = kahypar.Context()
    context.suppressOutput(True)
    context.setK(k)
    context.setEpsilon(0.03)
    context.loadINIconfiguration("test_env/lib/python3.10/site-packages/cotengra/pathfinders/kahypar_profiles/km1_kKaHyPar_sea20.ini")

    kahypar.partition(hypergraph, context)

    partitions = [[] for j in range(k)]
    for node in hypergraph.nodes():
        block = hypergraph.blockID(node)
        partitions[block].append(node)
    HyEdg_cut = 0
    for edge in edges:
        x = edge[0]
        y = edge[1]
        if hypergraph.blockID(x)!=hypergraph.blockID(y):
            HyEdg_cut+=1
    return partitions[0]

def sq_clifford_round(s,n,xy = False, circ = None):
    
    if xy:
        
        for j in range(n):
            r = random.choice(range(4))
            if r == 0:
                s.sqrt_x(j)
                if circ:
                    circ.add_gate(OpType.Rx,1/2,[j])
            elif r == 1:
                s.sqrt_x_dag(j)
                if circ:
                    circ.add_gate(OpType.Rx,-1/2,[j])
            elif r == 2:
                s.sqrt_y(j)
                if circ:
                    circ.add_gate(OpType.Ry,1/2,[j])
            elif r == 3:
                s.sqrt_y_dag(j)
                if circ:
                    circ.add_gate(OpType.Ry,-1/2,[j])
                
    if not xy:
    
        for j in range(n):
            r = random.choice(range(24))
            a,b = r // 4, r % 4
            
            
            if b == 1:
                s.x(j)
                if circ:
                    circ.add_gate(OpType.X,[j])
            elif b == 2:
                s.y(j)
                if circ:
                    circ.add_gate(OpType.Y,[j])
            elif b == 3:
                s.z(j)
                if circ:
                    circ.add_gate(OpType.Z,[j])

            if a == 1:
                s.h(j)
                if circ:
                    circ.add_gate(OpType.H,[j])
            elif a == 2:
                s.s(j)
                if circ:
                    circ.add_gate(OpType.S,[j])
            elif a == 3:
                s.s(j)
                if circ:
                    circ.add_gate(OpType.S,[j])
                s.h(j)
                if circ:
                    circ.add_gate(OpType.H,[j])
            elif a == 4:
                s.h(j)
                if circ:
                    circ.add_gate(OpType.H,[j])
                s.s(j)
                if circ:
                    circ.add_gate(OpType.S,[j])
            elif a == 5:
                s.h(j)
                if circ:
                    circ.add_gate(OpType.H,[j])
                s.s(j)
                if circ:
                    circ.add_gate(OpType.S,[j])
                s.h(j)
                if circ:
                    circ.add_gate(OpType.H,[j])
              
            

def circs(topology,xy = False, pytket = True):
    
    n, d = 2*len(topology[0]), len(topology)
    s = stim.TableauSimulator()
    if pytket:
        circ = Circuit(n)
    else:
        circ = None
        
    for layer in topology[:]:
        sq_clifford_round(s,n,xy = xy, circ = circ)
             
        for pair in layer:
            #The three gates below make up ZZMax
            s.cz(pair[0],pair[1])
            s.s(pair[0])
            s.s(pair[1])
            if circ:
                circ.add_gate(OpType.ZZMax,pair)
    
    sq_clifford_round(s,n,xy = xy, circ = circ)
    
    if pytket:
        return circ, s
    else:
        return s
    
    
    
def binaryMatrix(zStabilizers):
    """
        - Purpose: Construct the binary matrix representing the stabilizer states.
        - Inputs:
            - zStabilizers (array): The result of conjugating the Z generators on the initial state.
        Outputs:
            - binaryMatrix (array of size (N, 2N)): An array that describes the location of the stabilizers in the tableau representation.
    """
    N = len(zStabilizers)
    binaryMatrix = np.zeros((N,2*N))
    r = 0 # Row number
    for row in zStabilizers:
        c = 0 # Column number
        for i in row:
            if i == 3: # Pauli Z
                binaryMatrix[r,N + c] = 1
            if i == 2: # Pauli Y
                binaryMatrix[r,N + c] = 1
                binaryMatrix[r,c] = 1
            if i == 1: # Pauli X
                binaryMatrix[r,c] = 1
            c += 1
        r += 1

    return binaryMatrix

def getCutStabilizers(binaryMatrix, partition):
    """
        - Purpose: Return only the part of the binary matrix that corresponds to the qubits we want to consider for a bipartition.
        - Inputs:
            - binaryMatrix (array of size (N, 2N)): The binary matrix for the stabilizer generators.
            - cut (integer): Location for the cut.
        - Outputs:
            - cutMatrix (array of size (N, 2cut)): The binary matrix for the cut on the left.
    """
    cut = len(partition)
    N = len(binaryMatrix)
    cutMatrix = np.zeros((N,2*cut))

    for c in range(len(partition)):
        cutMatrix[:,c] = binaryMatrix[:,partition[c]]
        cutMatrix[:,cut+c] = binaryMatrix[:, N + partition[c]]
        
    return cutMatrix

def matToInts(mat):
    
    return [sum([(2**j)*int(mat[r][j]) for j in range(len(mat[r]))]) for r in range(len(mat))]
        
    
def gf2_rank(rows):
    """
    Find rank of a matrix over GF2.

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank


def array(zs):
    
    x = len(zs)
    y = len(zs[0])
    arr = [y*[0] for j in range(x)]
    
    
    for j in range(x):
        
        st = zs[j].__str__()[1:]
        
        for k in range(y):
            
            p=st[k]
        
            if p == 'X':
                arr[j][k] = 1
            elif p == 'Y':
                arr[j][k] = 2
            elif p =='Z':
                arr[j][k] = 3
    
    return np.array(arr)


def stabPurity(s,part):
    
    tableau = s.current_inverse_tableau() ** -1
    zs = [tableau.z_output(k) for k in range(len(tableau))]
    zs = array(zs)
    bm = binaryMatrix(zs)
    cs = getCutStabilizers(bm,part)
    ints = matToInts(cs)
        
    entropy = gf2_rank(ints) - len(part)
    rho2 = 2**(-entropy)

    return rho2