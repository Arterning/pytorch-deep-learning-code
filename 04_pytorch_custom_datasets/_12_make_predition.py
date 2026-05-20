# Download custom image
import requests
from pathlib import Path
import torchvision
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt




# 2. 将所有执行逻辑放入 main 块中
if __name__ == '__main__':

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # Setup custom image path
    custom_image_path = data_path / "04-pizza-dad.jpeg"

    # Download the image if it doesn't already exist
    if not custom_image_path.is_file():
        with open(custom_image_path, "wb") as f:
            # When downloading from GitHub, need to use the "raw" file link
            request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
            print(f"Downloading {custom_image_path}...")
            f.write(request.content)
    else:
        print(f"{custom_image_path} already exists, skipping download.")
    

    # Read in custom image
    custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))

    # Print out image data
    print(f"Custom image tensor:\n{custom_image_uint8}\n")
    print(f"Custom image shape: {custom_image_uint8.shape}\n")
    print(f"Custom image dtype: {custom_image_uint8.dtype}")




    # Load in custom image and convert the tensor values to float32
    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    custom_image = custom_image / 255. 

    # Print out image data
    print(f"Custom image tensor:\n{custom_image}\n")
    print(f"Custom image shape: {custom_image.shape}\n")
    print(f"Custom image dtype: {custom_image.dtype}")



    # Plot custom image
    plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
    plt.title(f"Image shape: {custom_image.shape}")
    plt.axis(False)
    plt.show()


    # Create transform pipleine to resize image
    custom_image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])

    # Transform target image
    custom_image_transformed = custom_image_transform(custom_image)

    # Print out original shape and new shape
    print(f"Original shape: {custom_image.shape}")
    print(f"New shape: {custom_image_transformed.shape}")