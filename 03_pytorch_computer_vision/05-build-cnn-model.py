# Import PyTorch
import torch
from torch import nn
from _01_prepare_data import train_data, test_data
from device_name import device
class_names = train_data.classes
from tqdm.auto import tqdm
from _02_data_loader import train_dataloader, test_dataloader
from helper_functions import print_train_time, accuracy_fn


# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, 
    hidden_units=10, 
    output_shape=len(class_names)).to(device)
model_2



def print_simple_batch():
    # Create sample batch of random numbers with same size as image batch
    images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
    test_image = images[0] # get a single image for testing
    print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
    print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
    print(f"Single image pixel values:\n{test_image}")

    torch.manual_seed(42)

    # Create a convolutional layer with same dimensions as TinyVGG 
    # (try changing any of the parameters and see what happens)
    conv_layer = nn.Conv2d(in_channels=3,
                        out_channels=10,
                        kernel_size=3,
                        stride=1,
                        padding=0) # also try using "valid" or "same" here 

    # Pass the data through the convolutional layer
    conv_layer(test_image) # Note: If running PyTorch <1.11.0, this will error because of shape issues (nn.Conv.2d() expects a 4d tensor as input) 


    # Add extra dimension to test image
    test_image.unsqueeze(dim=0).shape

    # Pass test image with extra dimension through conv_layer
    conv_layer(test_image.unsqueeze(dim=0)).shape



    torch.manual_seed(42)
    # Create a new conv_layer with different values (try setting these to whatever you like)
    conv_layer_2 = nn.Conv2d(in_channels=3, # same number of color channels as our input image
                            out_channels=10,
                            kernel_size=(5, 5), # kernel is usually a square so a tuple also works
                            stride=2,
                            padding=0)

    # Pass single image through new conv_layer_2 (this calls nn.Conv2d()'s forward() method on the input)
    conv_layer_2(test_image.unsqueeze(dim=0)).shape

    # Check out the conv_layer_2 internal parameters
    print(conv_layer_2.state_dict())

    # Get shapes of weight and bias tensors within conv_layer_2
    print(f"conv_layer_2 weight shape: \n{conv_layer_2.weight.shape} -> [out_channels=10, in_channels=3, kernel_size=5, kernel_size=5]")
    print(f"\nconv_layer_2 bias shape: \n{conv_layer_2.bias.shape} -> [out_channels=10]")


    # Print out original image shape without and with unsqueezed dimension
    print(f"Test image original shape: {test_image.shape}")
    print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")

    # Create a sample nn.MaxPoo2d() layer
    max_pool_layer = nn.MaxPool2d(kernel_size=2)

    # Pass data through just the conv_layer
    test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
    print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

    # Pass data through the max pool layer
    test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
    print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")




    torch.manual_seed(42)
    # Create a random tensor with a similar number of dimensions to our images
    random_tensor = torch.randn(size=(1, 1, 2, 2))
    print(f"Random tensor:\n{random_tensor}")
    print(f"Random tensor shape: {random_tensor.shape}")

    # Create a max pool layer
    max_pool_layer = nn.MaxPool2d(kernel_size=2) # see what happens when you change the kernel_size value 

    # Pass the random tensor through the max pool layer
    max_pool_tensor = max_pool_layer(random_tensor)
    print(f"\nMax pool tensor:\n{max_pool_tensor} <- this is the maximum value from random_tensor")
    print(f"Max pool tensor shape: {max_pool_tensor.shape}")




# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), 
                             lr=0.1)



torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
from engine import train_step, test_step, eval_model

train_time_start_model_2 = timer()

# Train and test model 
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                           end=train_time_end_model_2,
                                           device=device)

# Get model_2 results 
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)


print(model_2_results)













