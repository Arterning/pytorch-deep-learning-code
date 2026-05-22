# Import PyTorch
import torch
from torch import nn
from _01_prepare_data import train_data, test_data
from device_name import device
class_names = train_data.classes
from tqdm.auto import tqdm
from _02_data_loader import train_dataloader, test_dataloader
from helper_functions import print_train_time, accuracy_fn
from engine import make_predictions
# Import matplotlib for visualization
import matplotlib.pyplot as plt





def plot_preditions():

    import random
    random.seed(42)
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)


    # View the first test sample shape and label
    print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")


    # Make predictions on test samples with model 2
    pred_probs= make_predictions(model=model_2, 
                                data=test_samples)
    
    # Turn the prediction probabilities into prediction labels by taking the argmax()
    pred_classes = pred_probs.argmax(dim=1)

    # View first two prediction probabilities list
    pred_probs[:2]

    # Plot predictions
    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        # Create a subplot
        plt.subplot(nrows, ncols, i+1)

        # Plot the target image
        plt.imshow(sample.squeeze(), cmap="gray")

        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_names[pred_classes[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_names[test_labels[i]] 

        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        
        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g") # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r") # red text if wrong
        plt.axis(False);