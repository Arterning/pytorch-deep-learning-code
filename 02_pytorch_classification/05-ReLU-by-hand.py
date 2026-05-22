
# Standard PyTorch imports
import torch
from torch import nn
from data_setup import X_train, X_test, y_train, y_test
# Import matplotlib for visualization
import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary
from sklearn.datasets import make_circles


# Create ReLU function by hand 
def relu(x):
  return torch.maximum(torch.tensor(0), x) # inputs must be tensors



# Create a custom sigmoid function
def sigmoid(x):
  return 1 / (1 + torch.exp(-x))


def main():

  # Create a toy tensor (similar to the data going into our model(s))
  A = torch.arange(-10, 10, 1, dtype=torch.float32)

  # Visualize the toy tensor
  plt.plot(A);


  # Pass toy tensor through ReLU function
  relu(A)

  # Plot ReLU activated toy tensor
  plt.plot(relu(A))


  # Test custom sigmoid on toy tensor
  sigmoid(A)


  # Plot sigmoid activated toy tensor
  plt.plot(sigmoid(A));


if __name__ == "__main__":
    main()
