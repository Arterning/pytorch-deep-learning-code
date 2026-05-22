
# Standard PyTorch imports
import torch
from torch import nn
from _06_train_data import NUM_CLASSES, NUM_FEATURES, RANDOM_SEED, X_blob_train, X_blob_test, y_blob_train, y_blob_test
# Import matplotlib for visualization
import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary
from sklearn.datasets import make_circles


# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


from torch import nn

# # Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc



# Build model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(), # <- does our dataset require non-linear layers? (try uncommenting and see if the results change)
            nn.Linear(in_features=hidden_units, out_features=output_features), # how many classes are there?
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)


def make_predict(model_4):
    # Make prediction logits with model
    y_logits = model_4(X_blob_test.to(device))

    # Perform softmax calculation on logits across dimension 1 to get prediction probabilities
    y_pred_probs = torch.softmax(y_logits, dim=1) 
    print(y_logits[:5])
    print(y_pred_probs[:5])

    # Sum the first sample output of the softmax activation function 
    torch.sum(y_pred_probs[0])

    # Which class does the model think is *most* likely at the index 0 sample?
    print(y_pred_probs[0])
    print(torch.argmax(y_pred_probs[0]))


def train_loop():

    # Create an instance of BlobModel and send it to the target device
    model_4 = BlobModel(input_features=NUM_FEATURES, 
                        output_features=NUM_CLASSES, 
                        hidden_units=8).to(device)

    # Create loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_4.parameters(), 
                                lr=0.1) # exercise: try changing the learning rate here and seeing what happens to the model's performance



    # Perform a single forward pass on the data (we'll need to put it to the target device for it to work)
    model_4(X_blob_train.to(device))[:5]

    # How many elements in a single prediction sample?
    model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES 

    # Fit the model
    torch.manual_seed(42)

    # Set number of epochs
    epochs = 100

    

    for epoch in range(epochs):
        ### Training
        model_4.train()

        # 1. Forward pass
        y_logits = model_4(X_blob_train) # model outputs raw logits 
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels
        # print(y_logits)
        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_blob_train) 
        acc = accuracy_fn(y_true=y_blob_train,
                        y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model_4.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model_4(X_blob_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            # 2. Calculate test loss and accuracy
            test_loss = loss_fn(test_logits, y_blob_test)
            test_acc = accuracy_fn(y_true=y_blob_test,
                                    y_pred=test_pred)

        # Print out what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%") 


if __name__ == "__main__":
    train_loop()






# # Make predictions
# model_4.eval()
# with torch.inference_mode():
#     y_logits = model_4(X_blob_test)

# # View the first 10 predictions
# y_logits[:10]

# # Turn predicted logits in prediction probabilities
# y_pred_probs = torch.softmax(y_logits, dim=1)

# # Turn prediction probabilities into prediction labels
# y_preds = y_pred_probs.argmax(dim=1)

# # Compare first 10 model preds and test labels
# print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
# print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")




# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_4, X_blob_train, y_blob_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_4, X_blob_test, y_blob_test)




