###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
'''
In this script, we consume the randomness extracted (B) that is computationally indistinguishable from uniformly random as the randomness used for generating the validation set.
In the entropy accumulation framework, each round must be decided whether it is in the validation set with some probability gamma.
Each round is independent from each other.
For each round, we take 8 bits and convert it into a floating point number between 0 and 1.
If the floating point number is greater than some probability, we put it into the validation set.
We choose an overall validation probability of 0.59.
Instead of doing this in a single go, we do it in two passes.
This is because there is fluctuation to how many circuits we may be able to validate.
If we are aboe to validate fewer samples than expected, sequentially validating circuits will bias samples that appear first to be more likely verified.
If we use a pseudorandom permutation of the rounds, this breaks the independent assumption used by EAT.
As such, we first choose a conservative number of initial validation.
After that succeeds, we pick the remaining rounds.

Additionally, since B is also used to select random circuit measurement basis which is consumed from the top, we consume B from the bottom for validation selection.
B is large enough such that there is no overlap.
'''

import numpy as np

with open("../randomness/basis_randomness.txt", "r") as file:
    lines = file.readlines()

bits = ''
for line in lines[::-1][:8000]:
    bits += line[:-1]

gamma = 0.35
verify = []
for i in range(23651):
    bits_for_sample = bits[i * 8 : (i + 1) * 8]
    verify.append(int(bits_for_sample, 2) / 2**8 <= gamma)

np.save('initial_verification_set.npy', verify)


gamma = 0.24
second_verify = []
for i in range(23651, 2 * 23651):
    if verify[i % 23651] == 1:
        second_verify.append(False)
        continue
    bits_for_sample = bits[i * 8 : (i + 1) * 8]
    second_verify.append(int(bits_for_sample, 2) / 2**8 <= gamma)

np.save('second_verification_set.npy', second_verify)