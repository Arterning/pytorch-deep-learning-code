import torch

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
random_tensor, random_tensor.dtype


# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
zeros, zeros.dtype


# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
ones, ones.dtype


# Use torch.arange(), torch.range() is deprecated 
zero_to_ten_deprecated = torch.range(0, 10) # Note: this may return an error in the future
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
zero_to_ten