This foder contains code to run validation jobs on Aurora.
First, we need to generate the validation set by running `validation_set_generation.py`.
Then, we need to convert the experimental data into tensor network operands for contraction by running `get_operands.py`.
To run contraction, submit job to pbs job scheduler. The main contraction code is contained in main_torch.py.
To aggregate the contracted amplitudes to get the XEB score, run xeb.py. It shows the XEB score of the finished samples.