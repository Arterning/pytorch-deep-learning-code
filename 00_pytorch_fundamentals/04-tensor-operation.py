import torch

# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
tensor + 10

# Multiply it by 10
tensor * 10

# Subtract and reassign
tensor = tensor - 10
tensor

# Element-wise matrix multiplication
tensor * tensor

# Matrix multiplication
torch.matmul(tensor, tensor)

