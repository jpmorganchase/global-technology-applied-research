###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from cryptomite.raz import Raz
import pickle
import datetime

generating_B = False

if generating_B:
    n2 = 74207281
    n1 = n2 * 2
    m = 7400000
    raz = Raz(n1, m)
    with open('W1.pkl', 'rb') as f:
        W1_buffer = pickle.load(f)
    with open('W2.pkl', 'rb') as f:
        W2_buffer = pickle.load(f)
    W1_bits = [int(x) for x in list(f"{W1_buffer:0{2 * 74207281}b}")]
    W2_bits = [int(x) for x in list(f"{W2_buffer:0{74207281}b}")]
    print(f"{datetime.datetime.now()}: Extracting")
    B_bits = raz.extract(W1_bits, W2_bits)
    print(f"{datetime.datetime.now()}: Done")
    B_bits = ''.join(str(x) for x in B_bits)
    B_buffer = int(B_bits, 2)
    with open('B.pkl', 'wb') as f:
        B_buffer = pickle.dump(B_buffer, f)



extracting_seed_weak_good = True

if extracting_seed_weak_good:
    # Define experimental parameters
    n = 64
    L = 26857
    # n1 = 2 * 74207281 # Causes a bug when the other input is shorter than around 14, 16 M bits, probably due to out of memory. Not reproduced else where
    n1 = 2 * 43112609 # Weak source length
    n2 = n * L # Quantum randomness length
    m = 4093 # Seed output length
    raz = Raz(n1, m) # Initialize extractor
    # Loading weak source randomness
    with open('X.pkl', 'rb') as f:
        X_buffer = pickle.load(f)
    # Loading quantum randomness
    with open('Z.pkl', 'rb') as f:
        Z_buffer = pickle.load(f)
    X_bits = [int(x) for x in list(f"{X_buffer:0{n1}b}")]
    Z_bits = [int(x) for x in list(f"{Z_buffer:0{n2}b}")]
    # Extraction
    print(f"{datetime.datetime.now()}: Extracting")
    seed_bits = raz.extract(X_bits, Z_bits)
    print(f"{datetime.datetime.now()}: Done")
    seed_bits = ''.join(str(x) for x in seed_bits)
    seed_buffer = int(seed_bits, 2)
    # Saving seed
    with open('seed_weak_good.pkl', 'wb') as f:
        seed_buffer = pickle.dump(seed_buffer, f)



extracting_seed_quantum_good = False

if extracting_seed_quantum_good:

    n = 64
    L = 26857
    n_quantum_in = n * L

    valid_n1half = [
    3, 7, 15, 31, 63, 127, 255, 521, 1279, 2281, 3217, 4423, 23209,
    44497, 110503, 132049, 756839, 859433, 3021377, 6972593, 24036583,
    25964951, 30402457, 32582657, 42643801, 43112609, 74207281
    ]
    n_quantum_prelim = 2 * min(x for x in valid_n1half if 2 * x >= n_quantum_in)
    n1 = n_quantum_prelim
    n2 = n1 // 2
    m = 4093
    raz = Raz(n1, m)

    with open('X.pkl', 'rb') as f:
        X_buffer = pickle.load(f)
    
    with open('Z.pkl', 'rb') as f:
        Z_buffer = pickle.load(f)
    
    X_bits = [int(x) for x in list(f"{X_buffer:0{2 * 74207281}b}")][:n2]
    Z_bits = [int(x) for x in list(f"{Z_buffer:0{n1}b}")]

    print(f"{datetime.datetime.now()}: Extracting")
    seed_bits = raz.extract(Z_bits, X_bits)
    print(f"{datetime.datetime.now()}: Done")
    seed_bits = ''.join(str(x) for x in seed_bits)
    seed_buffer = int(seed_bits, 2)

    with open('seed_quantum_good.pkl', 'wb') as f:
        seed_buffer = pickle.dump(seed_buffer, f)


