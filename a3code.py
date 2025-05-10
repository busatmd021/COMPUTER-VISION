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

    # Store Loss for Each Batch
    batch_losses = []  

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
            loss_value = loss.item()
            batch_losses.append(loss_value)
        
    # Retrn Losses
    return batch_losses


## Define a Test Function
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



# ----------- PART 2 -----------