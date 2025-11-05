###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import torch
import numpy as np

# Setting tensors as contiguous does not help

def combine_dimensions(tensor: torch.Tensor, order):
    orig_shape = tensor.shape
    if len(order) <= 12:
        return tensor, order
    # order is a list of indices to permute the dimensions of the tensor
    # First, combine dimensions of tensors. Any dimensions that are consecutive in order will be combined into a single dimension.
    # For example, if order is [6,1,2,3,4,9,5,7,8,0], then dimension 1,2,3,4 will be combined into a single dimension, dimension 7,8 will be combined into a single dimension
    previous_index = order[0]
    new_order_not_compact = []
    all_combined_dimensions = []
    combined_dimensions = []
    for index in order[1:]:
        if previous_index + 1 == index:
            if len(combined_dimensions) == 0:
                combined_dimensions.append(previous_index)
            combined_dimensions.append(index)
        else:
            new_order_not_compact.append(previous_index)
            if len(combined_dimensions) > 0:
                all_combined_dimensions.append(combined_dimensions)
                combined_dimensions = []
        previous_index = index
    new_order_not_compact.append(previous_index)
    if len(combined_dimensions) > 0:
        all_combined_dimensions.append(combined_dimensions)
    # sort all_combined_dimensions by the first index of each combined dimension
    all_combined_dimensions.sort(key=lambda x: x[0], reverse=True)
    shape = list(tensor.shape)
    for combined_dimensions in all_combined_dimensions:
        combined_size = np.prod([orig_shape[i] for i in combined_dimensions])
        for index in combined_dimensions[::-1]:
            shape.pop(index)
        shape.insert(combined_dimensions[0], combined_size)
    tensor = tensor.reshape(shape)
    sorted_new_order_not_compact = sorted(new_order_not_compact)
    new_order = [sorted_new_order_not_compact.index(i) for i in new_order_not_compact]
    return tensor, new_order


def intermediate_permute(tensor: torch.Tensor, order):
    orig_shape = list(tensor.shape)
    orig_order = order.copy()
    tensor, order = combine_dimensions(tensor, order)
    combined_shape = list(tensor.shape)
    combined_order = order.copy()
    if len(order) <= 12:
        result = tensor.permute(order)
        result = result.reshape(np.array(orig_shape)[orig_order].tolist())
        return result
    new_shape, new_order = shape_order_cal_for_first_five_dim(tensor, order)
    tensor = tensor.reshape(new_shape)
    tensor = tensor.permute(new_order)
    combined_shape = [combined_shape[i] for i in combined_order[:5] + sorted(combined_order[5:])] # sorted is used since the remaining dimensions are not permuted yet
    new_shape = [np.prod(combined_shape[:5])]+ combined_shape[5:]
    tensor = tensor.reshape(new_shape)
    new_order = [0] + [sorted(combined_order[5:]).index(i) + 1 for i in combined_order[5:]]
    result = intermediate_permute(tensor, new_order)
    return result

def permute(tensor:torch.Tensor, order):
    return intermediate_permute(tensor, order).reshape([2] * len(order))



def shape_order_cal_for_first_five_dim(tensor: torch.Tensor, order):
    # Only permute the first five dimensions in order, other dimensions are combined
    # First, reshape the tensor to combine the remaining dimensions
    first_five_indices = order[:5]
    sorted_first_five_indices = sorted(first_five_indices)
    working_indices = first_five_indices[:]
    index = sorted_first_five_indices[0]
    if index == 0:
        new_shape = []
    else:
        combined_size = 1
        for size in tensor.shape[:index]:
            combined_size *= size
        new_shape = [combined_size]
        working_indices.append(index - 1)
    new_shape += [tensor.shape[index]]
    previous_index = index
    for index in sorted_first_five_indices[1:]:
        combined_size = 1
        for size in tensor.shape[previous_index+1:index]:
            combined_size *= size
        if combined_size != 1:
            new_shape += [combined_size]
            working_indices.append(index - 1)
        new_shape += [tensor.shape[index]]
        previous_index = index
    if previous_index != len(tensor.shape) - 1:
        combined_size = 1
        for size in tensor.shape[previous_index + 1:]:
            combined_size *= size
        new_shape += [combined_size]
        working_indices.append(previous_index + 1)
    sorted_working_indices = sorted(working_indices)
    new_order = [sorted_working_indices.index(i) for i in working_indices]
    return new_shape, new_order


failed = [214, 245, 301, 343, 361, 515, 551, 580, 581]
contraction_id = 0
contractions_per_tree = 635

def einsum(equation: str, input1: torch.Tensor, input2: torch.Tensor):
    global failed, contraction_id, contractions_per_tree
    # Tracking failed contractions and using only hack for the failed ones does not help
    if contraction_id % contractions_per_tree in failed:
        contraction_id += 1
        return hack_einsum(equation, input1, input2)
    else:
        contraction_id += 1
        result = torch.einsum(equation, input1, input2)
        return result


def fill_beffer_data(input):
    
    shape = list(input.shape); shape = [2] + shape
    buffer_tensors = torch.empty(shape, dtype = torch.complex32, device = input.device)
    buffer1, buffer2 = buffer_tensors[0], buffer_tensors[1]

    buffer1.real.copy_(input.real)
    buffer1.imag.copy_(-1*input.imag)
    buffer2.real.copy_(input.imag)
    buffer2.imag.copy_(input.real)
            
    return buffer1, buffer2, buffer_tensors



def hack_einsum(equation: str, input1: torch.Tensor, input2: torch.Tensor):
    global contraction_id
    dim_size = input1.shape[0]
    # Parsing inputs and output
    parts = equation.split(',')
    parts2 = parts[1].split('->')
    input1indices = parts[0]
    input2indices = parts2[0]
    outputindices = parts2[1]
    # Deciding index types
    batch_indices = []
    output_indices1 = []
    output_indices2 = []
    contracted_indices = []
    for input1index in input1indices:
        if input1index not in input2indices:
            if input1index not in outputindices:
                raise ValueError('Index {} is not in input2indices and not in outputindices'.format(input1index))
            output_indices1.append(input1index)
        else:
            if input1index not in outputindices:
                contracted_indices.append(input1index)
            else:
                batch_indices.append(input1index)
    for input2index in input2indices:
        if input2index not in input1indices:
            if input2index not in outputindices:
                raise ValueError('Index {} is not in input1indices and not in outputindices'.format(input2index))
            output_indices2.append(input2index)
    # print(batch_indices, output_indices1, output_indices2, contracted_indices)
    # Permuting inputs
    new_output_indices = ''
    input1_order = []
    all_indices1 = batch_indices + output_indices1 + contracted_indices
    for current_index in all_indices1:
        input1_order.append(input1indices.index(current_index))

    input1 = permute(input1, input1_order).reshape(dim_size ** len(batch_indices), dim_size ** len(output_indices1), -1)

    input2_order = []
    all_indices2 = batch_indices + contracted_indices + output_indices2
    for current_index in all_indices2:
        input2_order.append(input2indices.index(current_index))
    input2 = permute(input2, input2_order).reshape(dim_size ** len(batch_indices), dim_size ** len(contracted_indices), -1)
    # Contracting inputs
    output = torch.bmm(input1, input2).view([dim_size] * len(batch_indices + output_indices1 + output_indices2))
    # Permuting output
    for current_index in batch_indices + output_indices1 + output_indices2:
        new_output_indices += current_index
    output = permute(output, [new_output_indices.index(index) for index in outputindices])
    return output


def tensordot(input1, input2, *args, **kwargs):
    try:
        result = torch.tensordot(input1, input2, *args, **kwargs)
    except RuntimeError:
        raise RuntimeError(f'Torch tensordot failed with dimensions {len(input1.shape)}, {len(input2.shape)} and arguments {args}, {kwargs}')
    return result


def transpose(*args, **kwargs):
    try:
        result = permute(*args, **kwargs)
    except RuntimeError:
        raise RuntimeError(f'Torch transpose failed with dimensions {len(args[0].shape)}')
    return result


def stack(*args, **kwargs):
    try:
        result = torch.stack(*args, **kwargs)
    except RuntimeError:
        raise RuntimeError(f'Torch stack failed with dimensions {len(args[0].shape)}')
    return result


if __name__ == '__main__':
    import intel_extension_for_pytorch as ipex
    # import jax.numpy as jnp
    import numpy as jnp
    # Test the permute function
    shape = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    order = [6, 11, 13, 17, 21, 22, 0, 1, 2, 3, 4, 7, 10, 14, 16, 18, 20, 5, 8, 9, 12, 15, 19]
    # shape = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # order = [5,8,1,14,13,2,10,0,11,12,9,7,6,3,4]
    tensor = torch.randn(shape, device='xpu:0')
    # tensor = torch.randn(shape)
    permuted_tensor = permute(tensor, order)
    jax_input = jnp.array(tensor.cpu().numpy())
    jax_permuted_tensor = jnp.transpose(jax_input, axes=order)
    # print(permuted_tensor.shape, tensor.shape)  # Should print the shape of the permuted tensor
    print('Final correctness: ', jnp.allclose(jax_permuted_tensor, permuted_tensor.cpu().numpy()))  # Should print True if the results are close enough

    eqs = [
            'abcdefghijklmnopqrstuvwxy,kxczABCqaDset->abcdefghijlmnoprstuvwxyzABCD',
            'abcdefghijklmnopqrstuvwxyzAB,xfBntCoac->abcdefghijklmopqrstuvwyzABC',
            'abcdefghijklmnopqrstuvwxyzA,BuAn->abcdefghijklmopqrstvwxyzAB',
            'abcdefghijklmnopqrs,obslhrdte->acefgijkmnpqst',
            'abcdefghijklmnopqrstu,avkwmxypzsABeuCcoD->bcdfghijlnpqrtvwxyzABCD']
    contractions_per_tree = 100000
    for eq in eqs:
        tensor1, tensor2 = eq.split('->')[0].split(',')
        input1 = torch.randn([2]*len(tensor1), device='xpu:0', dtype=torch.float32) + 1j*torch.randn([2]*len(tensor1), device='xpu:0', dtype=torch.float32)
        input2 = torch.randn([2]*len(tensor2), device='xpu:0', dtype=torch.float32) + 1j*torch.randn([2]*len(tensor2), device='xpu:0', dtype=torch.float32)
        result = hack_einsum(eq, input1, input2)
        jax_input1 = jnp.array(input1.cpu().numpy())
        jax_input2 = jnp.array(input2.cpu().numpy())
        jax_result = jnp.einsum(eq, jax_input1, jax_input2)
        print(result.shape)  # Should print the shape of the result tensor
        partial_jax = jax_result
        partial_torch = result
        for _ in range(len(result.shape) - 3):
            partial_jax = partial_jax[1]
            partial_torch = partial_torch[1]
        print(partial_jax)
        print(partial_torch)
        print(jnp.allclose(jax_result, result.cpu().numpy(), atol=0.01, rtol=0.01))  # Should print True if the results are close enough