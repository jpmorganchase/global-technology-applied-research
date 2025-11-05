###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import json
from tqdm import tqdm

num_circuits = 11933

amplitudes = []
num_reused = 0
for circuit in tqdm(range(11933)):
    amplitudes_dir = f'/flare/CertRandomness/minzhaoliu/results/circuit_{circuit}_amplitudes.npy'
    slice_values = np.load(amplitudes_dir)
    assert np.sum(slice_values==0) == 0
    amplitudes.append(np.sum(slice_values).item())

probs = np.abs(amplitudes) ** 2
n = 64
xeb = 2**n * np.mean(probs) - 1
print(f'XEB: {xeb}.')


real = amplitudes.real.tolist()
imag = amplitudes.imag.tolist()
xeb_results = [[circuit for circuit in range(11933)], real, imag]
with open('../xeb_results.json', 'w') as f:
    json.dump(xeb_results, f)