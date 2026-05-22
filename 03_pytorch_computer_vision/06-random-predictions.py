# Import PyTorch
import torch
from torch import nn
from _01_prepare_data import train_data, test_data
from device_name import device
class_names = train_data.classes
from tqdm.auto import tqdm
from _02_data_loader import train_dataloader, test_dataloader
from _05_build_cnn_model import FashionMNISTModelV2
from helper_functions import print_train_time, accuracy_fn
from engine import make_predictions
# Import matplotlib for visualization
import matplotlib.pyplot as plt


from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "FashionMNISTModelV1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME



def train_model():

    torch.manual_seed(42)

    model_2 = FashionMNISTModelV2(input_shape=1, 
        hidden_units=10, 
        output_shape=len(class_names)).to(device)
    

    if Path(MODEL_SAVE_PATH).is_file():
        print(f"Loading model to: {MODEL_SAVE_PATH}")
        model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
        return model_2

        
    # Setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_2.parameters(), 
                                lr=0.1)


    torch.manual_seed(42)

    # Measure time
    from timeit import default_timer as timer
    from engine import train_step, test_step, eval_model

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

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_2.state_dict(), # only saving the state_dict() only saves the models learned parameters
            f=MODEL_SAVE_PATH)

    return model_2



def plot_preditions(model_2):

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
    print(pred_probs[:2])
    print(pred_classes)

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

        # 关键：保存成图片，无桌面环境不要 plt.show()
        plt.savefig(f"images/{i}_my_plot.png", dpi=300, bbox_inches='tight')
        plt.close()  # 释放内存

        print(f"图片已保存为：{i}_my_plot.png")




# 2. 将所有执行逻辑放入 main 块中
if __name__ == '__main__':


    model_2 = train_model()

    
    
    plot_preditions(model_2)