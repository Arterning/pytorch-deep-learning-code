"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary
import data_setup, engine, utils
from helper_functions import plot_loss_curves


# 2. 将所有执行逻辑放入 main 块中
if __name__ == '__main__':

    # Setup hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # Setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create image size (from Table 3 in the ViT paper)
    IMG_SIZE = 224

    # Create transform pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    print(f"Manually created transforms: {manual_transforms}")



    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,
        batch_size=BATCH_SIZE
    )

    print("class_names", class_names)

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

    model = torchvision.models.efficientnet_b0(weights=weights).to(device)


    # Print a summary using torchinfo (uncomment for actual output)
    summary(model=model, 
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)
    
    # # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
    summary(model, 
            input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    print("summary", summary)




    # # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)

    # Set the random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Setup training and save the results
    results = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=5,
                        device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    plot_loss_curves(results)

    plt.savefig("results.png", dpi=300, bbox_inches='tight')
    plt.close() 


    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="06_pytorch_transfer_learning.pth")