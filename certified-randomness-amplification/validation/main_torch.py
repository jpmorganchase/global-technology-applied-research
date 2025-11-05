###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
gpus_per_controller = 12
control_color = rank % (gpus_per_controller + 2)
control_key = rank // (gpus_per_controller + 2)
control_subcomm = comm.Split(control_color, control_key)
color = rank // (gpus_per_controller + 2)
key = rank % (gpus_per_controller + 2)
subcomm = comm.Split(color, key)

import time
import sys
import os
import pickle
import numpy as np
from math import ceil
import datetime

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

with suppress_stdout_stderr():
    import torch
    import intel_extension_for_pytorch as ipex
    device = f'xpu:0'

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import autoray
    import hack_torch
    autoray.register_backend(torch.Tensor, 'hack_torch')
    with open(f"tree.pkl", "rb") as f:
        tree = pickle.load(f)
    nslices = tree.nslices




def scheduler_main(task_ids):

    available_controllers = [controller for controller in range(1, control_subcomm.Get_size())]
    requests = []
    dispatched_tasks = []
    dispatched_controllers = []
    requests_start_time = []
    prev_timeout_check_time = time.time()

    while True:

        if time.time() - prev_timeout_check_time > 10:
            timeout_locations = np.where(time.time() - np.array(requests_start_time) > 300)[0]
            prev_timeout_check_time = time.time()
            for timeout_location in sorted(timeout_locations, reverse=True):
                # pop from large locations to small. Otherwise, the indices will shift and cause an error.
                task_id = dispatched_tasks.pop(timeout_location)
                timeout_controller = dispatched_controllers.pop(timeout_location)
                requests_start_time.pop(timeout_location)
                requests.pop(timeout_location)
                task_ids.insert(0, task_id)
                print(f'Timeout detected for task {task_id} on controller {timeout_controller}.')

        test_result = MPI.Request.Testany(requests)
        completed_location = test_result[0]
        if completed_location >= 0:
            task_id = dispatched_tasks.pop(completed_location)
            controller = dispatched_controllers.pop(completed_location)
            requests_start_time.pop(completed_location)
            requests.pop(completed_location)
            available_controllers.append(controller)

        if len(available_controllers) != 0 and len(task_ids) != 0:
            controller = available_controllers.pop(0)
            task_id = task_ids.pop(0)
            control_subcomm.isend(task_id, dest=controller, tag=63843)
            dispatched_tasks.append(task_id)
            dispatched_controllers.append(controller)
            requests_start_time.append(time.time())
            requests.append(control_subcomm.irecv(source=controller, tag=32341))
        



def collection_main(slices_per_sub_task, tasks_per_circuit, circuit_status):

    start = time.time()
    previous_backup_time = time.time()
    previous_print_time = time.time()
    num_completed = 0
    task_id = None
    active_amplitudes = {}

    buffers = [np.zeros([slices_per_sub_task * gpus_per_controller], dtype=np.complex64) for _ in range(control_subcomm.Get_size() - 1)]
    buffer_requests = [None for _ in range(control_subcomm.Get_size() - 1)]
    for controller in range(control_subcomm.Get_size() - 1):
        buffer_requests[controller] = comm.Irecv([buffers[controller], MPI.C_FLOAT_COMPLEX], (controller + 1) * (gpus_per_controller + 2), tag=7824)

    while True:
    
        if time.time() - previous_print_time > 5 or num_completed == total_tasks:
            previous_print_time = time.time()
            dt = datetime.datetime.now()
            active_circuits = list(active_amplitudes.keys())
            slices_completed = np.sum([np.sum(~np.isnan(active_amplitudes[active_circuit])) for active_circuit in active_circuits])
            print(task_id, num_completed, dt.day, dt.hour, dt.minute, dt.second, active_amplitudes.keys(), slices_completed, flush=True)
            sys.stdout.flush()
            if time.time() - previous_backup_time > 60:
                previous_backup_time = time.time()
                active_circuits = list(active_amplitudes.keys())
                for circuit in active_circuits:
                    amplitudes_dir = f'results/circuit_{circuit}_amplitudes.npy'
                    backup_amplitudes_dir = f'results/circuit_{circuit}_amplitudes_backup.npy'
                    # backup previously saved files into filename_backup.npy
                    if os.path.exists(amplitudes_dir):
                        amplitudes_backup = np.load(amplitudes_dir)
                        np.save(backup_amplitudes_dir, amplitudes_backup)
                    np.save(amplitudes_dir, active_amplitudes[circuit].reshape(-1)[:nslices])
                    # Update circuit_status and clear memory of active_amplitudes if circuit is fully computed
                    if (~np.isnan(active_amplitudes[circuit].reshape(-1)[:nslices])).all():
                        del active_amplitudes[circuit]
                        circuit_status[circuit] = 1
                        np.save(status_dir, circuit_status)
                    elif circuit_status[circuit] == 0:
                        circuit_status[circuit] = 2
                        np.save(status_dir, circuit_status)
        if num_completed == total_tasks:
            active_circuits = list(active_amplitudes.keys())
            for circuit in active_circuits:
                amplitudes_dir = f'results/circuit_{circuit}_amplitudes.npy'
                backup_amplitudes_dir = f'results/circuit_{circuit}_amplitudes_backup.npy'
                # backup previously saved files into filename_backup.npy
                if os.path.exists(amplitudes_dir):
                    amplitudes_backup = np.load(amplitudes_dir)
                    np.save(backup_amplitudes_dir, amplitudes_backup)
                np.save(amplitudes_dir, active_amplitudes[circuit].reshape(-1)[:nslices])
                # Update circuit_status and clear memory of active_amplitudes if circuit is fully computed
                if (~np.isnan(active_amplitudes[circuit].reshape(-1)[:nslices])).all():
                    del active_amplitudes[circuit]
                    circuit_status[circuit] = 1
                    np.save(status_dir, circuit_status)
                elif circuit_status[circuit] == 0:
                    circuit_status[circuit] = 2
                    np.save(status_dir, circuit_status)
            print('All tasks completed.')
            sys.stdout.flush()
            MPI.Finalize()
            quit()

        test_result = MPI.Request.Testany(buffer_requests)
        controller = test_result[0]
        if controller < 0:
            continue
            
        while not comm.Iprobe(source=(controller + 1) * (gpus_per_controller + 2), tag=4821):
            pass
        task_id = comm.recv(source=(controller + 1) * (gpus_per_controller + 2), tag=4821)
        task_amplitudes = np.copy(buffers[controller])
        circuit = task_id // tasks_per_circuit
        task_in_circuit = task_id % tasks_per_circuit
        # see if it is in active_amplitudes
        if circuit not in active_amplitudes:
            amplitudes_dir = f'results/circuit_{circuit}_amplitudes.npy'
            circuit_amplitudes = np.full((tasks_per_circuit * slices_per_sub_task * gpus_per_controller), np.nan, dtype=np.complex64)
            if os.path.exists(amplitudes_dir):
                circuit_amplitudes[:nslices] = np.load(amplitudes_dir)
            circuit_amplitudes = circuit_amplitudes.reshape((tasks_per_circuit, slices_per_sub_task * gpus_per_controller))
            active_amplitudes[circuit] = circuit_amplitudes
        active_amplitudes[circuit][task_in_circuit, :] = task_amplitudes

        num_completed += 1
        
        buffers[controller].fill(np.nan)
        task_ids[controller] = -1
        buffer_requests[controller] = comm.Irecv([buffers[controller], MPI.C_FLOAT_COMPLEX], (controller + 1) * (gpus_per_controller + 2), tag=7824)




def control_main(slices_per_sub_task):

    def assign(task_id):
        sub_task_begin = task_id * gpus_per_controller
        for i in range(gpus_per_controller):
            sub_task_id = sub_task_begin + i
            subcomm.isend(sub_task_id, dest=i+2, tag=44762)
        results = np.zeros([slices_per_sub_task * gpus_per_controller], dtype=np.complex64)
        requests = []
        for i in range(gpus_per_controller):
            requests.append(subcomm.Irecv([results[i * slices_per_sub_task : (i + 1) * slices_per_sub_task], MPI.C_FLOAT_COMPLEX], source=i+2, tag=28344))
        MPI.Request.Waitall(requests)
        return results

    requests = []
    while True:
        # Other ranks will receive tasks and perform computations
        task_id = control_subcomm.recv(source=0, tag=63843)
        results = assign(task_id)
        MPI.Request.Waitall(requests)
        task_id_request = comm.isend(task_id, dest=1, tag=4821)
        results_request = comm.Isend([results, MPI.C_FLOAT_COMPLEX], dest=1, tag=7824)
        done_request = control_subcomm.isend(True, dest=0, tag=32341)
        requests = [results_request, task_id_request, done_request]




def rank_main(slices_per_sub_task):

    slices_per_task = slices_per_sub_task * gpus_per_controller
    tasks_per_circuit = ceil(nslices / slices_per_task)
    sub_tasks_per_circuit = tasks_per_circuit * gpus_per_controller

    cpu_operands_dict = {}

    def contract(sub_task_id):
        circuit = sub_task_id // sub_tasks_per_circuit
        task_in_circuit = sub_task_id % sub_tasks_per_circuit
        slice_begin = task_in_circuit * slices_per_sub_task
        slice_end = (task_in_circuit + 1) * slices_per_sub_task
        operands = [torch.tensor(operand, dtype=torch.complex64, device=device) for operand in all_operands[circuit]]

        with torch.no_grad():
            results = []
            for idx in range(slice_begin, slice_end):
                if idx >= nslices:
                    results.append(0.0 + 0.0j)
                else:
                    results.append(tree.contract_core(tree.slice_arrays(operands, idx), prefer_einsum=True).cpu())
            return np.array(results, dtype=np.complex64)

    while True:
        # Other ranks will receive tasks and perform computations
        sub_task_id = subcomm.recv(source=0, tag=44762)
        results = contract(sub_task_id)
        subcomm.Send([results, MPI.C_FLOAT_COMPLEX], dest=0, tag=28344)





slices_per_sub_task = 512
num_circuits = 11933

all_operands = np.full([num_circuits, 636, 2, 2], np.nan, dtype=np.complex64)
if rank == 0:
    dt = datetime.datetime.now()
    print('Loading operands: ', dt.day, dt.hour, dt.minute, dt.second, flush=True)
    all_operands = []
    for circuit in range(num_circuits):
        all_operands.append(np.load(f'operands/operands_{circuit}.npy'))
    all_operands = np.array(all_operands, dtype=np.complex64)
    dt = datetime.datetime.now()
    print('Broadcasting operands: ', dt.day, dt.hour, dt.minute, dt.second, flush=True)
comm.Bcast([all_operands, MPI.C_FLOAT_COMPLEX])
comm.barrier()
assert np.isnan(all_operands).sum() == 0


if rank in [0, 1]:

    if rank == 0:
        dt = datetime.datetime.now()
        print('Initializing jobs: ', dt.day, dt.hour, dt.minute, dt.second, flush=True)
    slices_per_task = slices_per_sub_task * gpus_per_controller
    tasks_per_circuit = ceil(nslices / slices_per_task)

    num_circuits = 11933

    # circuit_status is an array that has value 0 if the circuit has not been computed, 1 if it has been computed, and 2 if it has been partially computed
    status_dir = f'results/circuit_status.npy'
    if os.path.exists(status_dir):
        circuit_status = np.load(status_dir)
    else:
        circuit_status = np.zeros(num_circuits, dtype=int)
    completed = np.where(circuit_status == 1)[0]
    partially_completed = np.where(circuit_status == 2)[0]
    if rank == 0:
        print('circuit_status: ', circuit_status, flush=True)
        print('partially completed: ', partially_completed, flush=True)
        sys.stdout.flush()
    
    task_completed_array = np.zeros([num_circuits, tasks_per_circuit], dtype=bool)
    task_completed_array[completed, :] = True
    for circuit in partially_completed:
        # See if saved results exists
        amplitudes_dir = f'results/circuit_{circuit}_amplitudes.npy'
        if os.path.exists(amplitudes_dir):
            circuit_amplitudes = np.full((tasks_per_circuit * slices_per_sub_task * gpus_per_controller), np.nan, dtype=np.complex64)
            circuit_amplitudes[:nslices] = np.load(amplitudes_dir)
            circuit_amplitudes = circuit_amplitudes.reshape((tasks_per_circuit, slices_per_sub_task * gpus_per_controller))
            for task_in_circuit in range(tasks_per_circuit):
                task_amplitudes = circuit_amplitudes[task_in_circuit]
                if (~np.isnan(task_amplitudes)).all():
                    task_completed_array[circuit, task_in_circuit] = True

    task_ids = np.where(~task_completed_array.reshape(-1))[0]
    total_tasks = len(task_ids)

    if rank == 0:
        print('circuit_stats: ', circuit_status, flush=True)
        dt = datetime.datetime.now()
        print('Initializing scheduler: ', dt.day, dt.hour, dt.minute, dt.second, flush=True)
        sys.stdout.flush()
        task_ids = list(task_ids)
        scheduler_main(task_ids)
    else:
        collection_main(slices_per_sub_task, tasks_per_circuit, circuit_status)

elif control_color == 0:
    control_main(slices_per_sub_task)

elif control_color != 1:
    rank_main(slices_per_sub_task)


MPI.Finalize()