import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from _08_build_model_v0 import TinyVGG
from _09_train_and_test import train
from _10_train_model_v0 import plot_loss_curves


# 2. 将所有执行逻辑放入 main 块中
if __name__ == '__main__':

    device = "cpu"

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Create training transform with TrivialAugment
    train_transform_trivial_augment  = transforms.Compose([ 
        transforms.Resize((64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
    ])

    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load and transform data
    train_data_augmented = datasets.ImageFolder(root=train_dir, transform=train_transform_trivial_augment)
    test_data_simple = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Setup batch size and number of workers 
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

    test_dataloader_simple = DataLoader(test_data_simple, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=NUM_WORKERS)


        
    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 5

    # Recreate an instance of TinyVGG
    model_1 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                    hidden_units=10, 
                    output_shape=len(train_data_augmented.classes)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model_1
    model_1_results = train(model=model_1, 
                            train_dataloader=train_dataloader_augmented,
                            test_dataloader=test_dataloader_simple,
                            optimizer=optimizer,
                            loss_fn=loss_fn, 
                            epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")


    plot_loss_curves(model_1_results)