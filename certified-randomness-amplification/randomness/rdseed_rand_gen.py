###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from rdrand import RdSeedom
from tqdm import tqdm
import pickle

rdseed = RdSeedom()

print('W1')
W1 = rdseed.getrandbits(2*74207281)
print('discarded')
discarded = rdseed.getrandbits(2*74207281)
print('W2')
W2 = rdseed.getrandbits(74207281)
print('X')
X = rdseed.getrandbits(2*43112609)
print('circuit randomness')
circuit_randomness_list = []
print()

for _ in tqdm(range(100000)):
    circuit_randomness = rdseed.getrandbits(3 * (64 * 9 + 40))
    circuit_randomness_list.append(circuit_randomness)

with open('W1.pkl', 'wb') as f:
    pickle.dump(W1, f)

with open('discarded.pkl', 'wb') as f:
    pickle.dump(discarded, f)

with open('W2.pkl', 'wb') as f:
    pickle.dump(W2, f)

with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('circuit_randomness_list.pkl', 'wb') as f:
    pickle.dump(circuit_randomness_list, f)