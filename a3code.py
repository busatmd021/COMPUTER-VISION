# ASSIGNMENT 3 CODE TO SUPPORT NOTEBOOK

# ----------- Import Libraries -----------
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# ----------- PART 1 -----------
def train(dataloader, model, loss_fn, optimizer):
    """
    In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the models' parameters.

    Args:
        Dataloader (DataLoader): The training data loader.
        Model (nn.Module): The neural network to train.
        Loss_fn (function): The loss function used to compute the error.
        Optimiser (torch.optim.Optimizer): The optimiser used to update the model parameters.
    """
    # Get CPU or GPU Device for Training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Total Number of Samples in the Dataset
    size = len(dataloader.dataset)  

    # Set the Model to Training Mode (Enables Dropout, Batchnorm, etc.)
    model.train()  

    for batch, (X, y) in enumerate(dataloader):
        # Move Input Data & Labels to the Selected Device (CPU or GPU)
        X, y = X.to(device), y.to(device)

        # Forward Pass: Compute Model Predictions & Loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation: Compute Gradients & Update Parameters
        optimizer.zero_grad()   # Clear Previous Gradients
        loss.backward()         # Backpropagate the Error
        optimizer.step()        # Update Model Parameters

        # Print Loss Every 100 Batches for Progress Tracking
        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """
    Evaluate the Model on the Test Dataset.

    Args:
        Dataloader (DataLoader): The test data loader.
        Model (nn.Module): The trained model to evaluate.
        Loss_fn (function): The loss function used to compute error.

    Prints:
        Average loss and overall accuracy on the test set.
    """
    # Get CPU or GPU Device for Training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Total Number of Test Samples
    size = len(dataloader.dataset)      

    # Total Number of Batches 
    num_batches = len(dataloader)     

    # Set the Model to Evaluation Mode (disables dropout, etc.)   
    model.eval()        

    # Initialise Total Loss & Correct Prediction Counter                 
    test_loss, correct = 0, 0            

    # Disable Gradient Computation for Inference
    with torch.no_grad():              
        for X, y in dataloader:
            # Move Data to the Appropriate Device
            X, y = X.to(device), y.to(device) 

            # Make Predictions      
            pred = model(X)        

            # Accumulate Loss                 
            test_loss += loss_fn(pred, y).item()    

            # Count Correct Predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  

    # Average Loss Over all Batches
    test_loss /= num_batches      

    # Accuracy: Correct Predictions / Total Samples     
    correct /= size                     

    # Display Evaluation Results
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Return Validation Loss
    return test_loss


def train_model(model, train_dataloader, test_dataloader, epochs = 10, lr = 1):
    """
    Train and test a model using the provided data loaders, loss function, and optimiser.

    Args:
        Model (torch.nn.Module): The neural network model to be trained.
        Train_dataloader (DataLoader): The DataLoader for the training dataset.
        Test_dataloader (DataLoader): The DataLoader for the testing dataset.
        Epochs (int, optional): The number of training epochs (default is 10).
        Lr (float, optional): The learning rate for the optimizer (default is 1).

    Returns:
        List: A list containing the validation loss after each epoch.
    """
    # Define the Loss Function & the Optimiser (learning Rate)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)   

    # Prepare to Store Loss
    val_loss = []

    # Train & Test the Model
    for t in range(epochs):  
        print(f"Epoch {t+1}\n-------------------------------")

        # Train and collect batch losses
        train(train_dataloader, model, loss_fn, optimizer)

        # Run testing
        loss = test(test_dataloader, model, loss_fn)
        val_loss.append(loss)
    print("Done!")

    return val_loss


def display_loss(title, x_label, y_label, val_loss, epochs = 10):
    """
    Display the loss over epochs as a line plot.

    Args:
        Title (str): The title of the plot.
        X_label (str): The label for the x-axis.
        Y_label (str): The label for the y-axis.
        Val_loss (list): A list of validation losses for each epoch.
        Epochs (int, optional): The number of epochs (default is 10).
    """
    # Create a Figure for the Plot with a Specified Size
    plt.figure(figsize=(8, 5))

    # Plot the Validation Loss Over the Epochs
    plt.plot(range(1, epochs + 1), val_loss, marker='o')

    # Label the Axes & Add a Title
    plt.xlabel(x_label, fontweight="bold")
    plt.ylabel(y_label, fontweight="bold")
    plt.title(title, fontweight="bold")

    # Display Gridlines for Better Readability
    plt.grid(True)

    # Show the Plot
    plt.show()



# ----------- PART 2 -----------