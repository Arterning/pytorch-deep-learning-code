import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from _08_build_model_v0 import TinyVGG
from _09_train_and_test import train



def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


# 2. 将所有执行逻辑放入 main 块中
if __name__ == '__main__':

    device = "cpu"

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Create simple transform
    simple_transform = transforms.Compose([ 
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load and transform data
    train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
    test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

    # Setup batch size and number of workers 
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

    # Create DataLoader's
    train_dataloader_simple = DataLoader(train_data_simple, 
                                         batch_size=BATCH_SIZE, 
                                         shuffle=True, 
                                         num_workers=NUM_WORKERS)

    test_dataloader_simple = DataLoader(test_data_simple, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=NUM_WORKERS)

    print(train_dataloader_simple, test_dataloader_simple)


        
    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 5

    # Recreate an instance of TinyVGG
    model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                    hidden_units=10, 
                    output_shape=len(train_data_simple.classes)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model_0 
    model_0_results = train(model=model_0, 
                            train_dataloader=train_dataloader_simple,
                            test_dataloader=test_dataloader_simple,
                            optimizer=optimizer,
                            loss_fn=loss_fn, 
                            epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")


    plot_loss_curves(model_0_results)