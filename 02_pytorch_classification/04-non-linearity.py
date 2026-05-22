# Standard PyTorch imports
import torch
from torch import nn
from data_setup import X_train, X_test, y_train, y_test
# Import matplotlib for visualization
import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary
from sklearn.datasets import make_circles


# Build model with non-linear activation function
from torch import nn

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# # Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

def train_loop():

    model_3 = CircleModelV2().to(device)
    print(model_3)

    # Setup loss and optimizer 
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)


    # Fit the model
    torch.manual_seed(42)
    epochs = 1000

    for epoch in range(epochs):
        # 1. Forward pass
        y_logits = model_3(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy_fn(y_true=y_train, 
                        y_pred=y_pred)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model_3.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_3(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
            # 2. Calculate loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                    y_pred=test_pred)

            # Print out what's happening
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

    return model_3

def make_predict(model_3):
    # Make predictions
    model_3.eval()
    with torch.inference_mode():
        y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
    print(y_preds[:10])

def plot_predict(model_3):
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity
    plt.savefig(f"plot.png", dpi=300, bbox_inches='tight')
    plt.close()  # 释放内存


if __name__ == "__main__":
    model_3 = train_loop()
    make_predict(model_3)
    plot_predict(model_3)





def make_samples():
    n_samples = 1000

    X, y = make_circles(n_samples=1000,
        noise=0.03,
        random_state=42,
    )

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)


    # Convert to tensors and split into train and test sets
    import torch
    from sklearn.model_selection import train_test_split

    # Turn data into tensors
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2,
                                                        random_state=42
    )

    X_train[:5], y_train[:5]

