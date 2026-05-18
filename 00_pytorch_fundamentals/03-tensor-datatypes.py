import torch

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device


# Getting information from tensors

# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU


# Change the datatype of a tensor
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
tensor.dtype

# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
tensor_float16

# Create an int8 tensor
tensor_int8 = tensor.type(torch.int8)
tensor_int8


