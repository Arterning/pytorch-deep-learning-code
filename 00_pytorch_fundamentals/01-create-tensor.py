import torch

# Scalar
scalar = torch.tensor(7)
scalar


# Vector
vector = torch.tensor([7, 7])
vector


# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print("TENSOR", TENSOR)
print("dim", TENSOR.ndim, "shape", TENSOR.shape)