###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import pickle
from tqdm import tqdm

with open('B.pkl', 'rb') as f:
    B_buffer = pickle.load(f)

B_bitstring = f"{B_buffer:0{7400000}b}"

with open('basis_randomness.txt', 'w') as f:
    for i in tqdm(range(115625)):
        line = B_bitstring[i * 64 : (i + 1) * 64] + '\n'
        f.write(line)