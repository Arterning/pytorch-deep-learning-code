from sklearn.datasets import make_circles

def make_samples():
    # Make 1000 samples 
    n_samples = 1000

    # Create circles
    X, y = make_circles(n_samples,
                        noise=0.03, # a little bit of noise to the dots
                        random_state=42) # keep random state so we get the same values

    print(f"First 5 X features:\n{X[:5]}")
    print(f"\nFirst 5 y labels:\n{y[:5]}")


    # Make DataFrame of circle data
    import pandas as pd
    circles = pd.DataFrame({"X1": X[:, 0],
        "X2": X[:, 1],
        "label": y
    })
    circles.head(10)
    # Check different labels
    circles.label.value_counts()

    # Visualize with a plot
    import matplotlib.pyplot as plt
    plt.scatter(x=X[:, 0], 
                y=X[:, 1], 
                c=y, 
                cmap=plt.cm.RdYlBu)
    
    plt.savefig(f"circle.png", dpi=300, bbox_inches='tight')
    plt.close()  # 释放内存


    # Check the shapes of our features and labels
    X.shape, y.shape


    # Turn data into tensors
    # Otherwise this causes issues with computations later on
    import torch
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    # View the first five samples
    X[:5], y[:5]



    # Split data into train and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, # 20% test, 80% train
                                                        random_state=42) # make the random split reproducible

    len(X_train), len(X_test), len(y_train), len(y_test)




if __name__ == "__main__":
    make_samples()





