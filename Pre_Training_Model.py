# ------------------------ IMPORTS ------------------------
# Standard Library Imports
import os

# PyTorch Core Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# Torchvision Imports (Datasets & Transforms)
from torchvision import transforms
from torchvision.datasets import Cityscapes

# Metrics & Utilities
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.tensorboard import SummaryWriter

# Progress Bar Utility
from tqdm import tqdm



# ---------------------- CONFIG ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 50
MODEL_SAVE_PATH = "./Models/pretrained_weights.pth"
LOG_DIR = "./logs"
TRAIN_DIR = "./CityScapes-Dataset"
TEST_DIR = "./CityScapes-Dataset"
NUM_CLASSES = 19



# ---------------------- DATA PREPERATION -----------------
def get_data_loader(data_dir, batch_size, split='train'):
    """Prepare Cityscapes dataset and dataloader for the specified split."""
    
    # Define Image Transformations: Convert Images to Tensors & Normalise
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet Means
                             std=[0.229, 0.224, 0.225]),  # ImageNet stds
    ])

    # Load the Cityscapes Dataset with Semantic Segmentation Labels
    dataset = Cityscapes(
        root=data_dir,         # Path to Dataset Root Folder
        split=split,           # Use the Specified Split ('train', 'val', 'test')
        mode='fine',           # Use Fine Annotations
        target_type='semantic',# Semantic Segmentation Masks
        transform=transform,   # Apply Transforms to Input Images
        target_transform=transform # Apply Transforms to Target Masks
    )

    # Create DataLoader for Batch Loading & Shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),  # Shuffle ONLY if Training Data
        num_workers=4,               # Use Multiple Subprocesses for Data Loading
        pin_memory=True              # Pin Memory for Faster GPU Transfers
    )

    return dataloader



# ---------------------- CNN MODEL DEFINITION --------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # Upsample x to Match Skip Spatial Size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate Along Channels
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.gn1(self.conv1(x)))
        x = self.relu2(self.gn2(self.conv2(x)))
        
        # Return th Result
        return x
    

class BetterSegNetwork(nn.Module):
    def __init__(self, n_class=19):
        super(BetterSegNetwork, self).__init__()

        # ----------------------------- Encoder -------------------------------
        # Stage 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.gn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.gn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        
        # Pooling Layer 1/2 Scale
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True) # --> 1/2

        # Residual Conv for Stage 1 (input channels 3 → 64)
        self.res_conv1 = nn.Conv2d(3, 64, kernel_size=1)  

        # Stage 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.gn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)

        # Pooling Layer 1/2 Scale
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True) # --> 1/4

        # Residual Conv for Stage 2 (input channels 64 → 128)
        self.res_conv2 = nn.Conv2d(64, 128, kernel_size=1)

        # Stage 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.gn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.gn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.gn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)

        # Pooling Layer 1/2 Scale
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True) # --> 1/8

        # Residual Conv for Stage 3 (input channels 128 → 256)
        self.res_conv3 = nn.Conv2d(128, 256, kernel_size=1)

        # Stage 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.gn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.drop4_1 = nn.Dropout2d(0)

        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.gn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.drop4_2 = nn.Dropout2d(0)

        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.gn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.drop4_3 = nn.Dropout2d(0)

        # Pooling Layer 1/2 Scale
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True) # --> 1/16

        # Residual Conv for Stage 4 (input channels 256 → 512)
        self.res_conv4 = nn.Conv2d(256, 512, kernel_size=1)

        # Stage 5
        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=2, dilation=2)  # Dilation = 2 expands receptive field
        self.gn5_1 = nn.BatchNorm2d(1024)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.drop5_1 = nn.Dropout2d(0)

        self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=2, dilation=2)
        self.gn5_2 = nn.BatchNorm2d(1024)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.drop5_2 = nn.Dropout2d(0)

        self.conv5_3 = nn.Conv2d(1024, 1024, 3, padding=2, dilation=2)
        self.gn5_3 = nn.BatchNorm2d(1024)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.drop5_3 = nn.Dropout2d(0)

        # Pooling Layer 1/2 Scale
        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True) 

        # Residual Conv for Stage 5 (input channels 512 → 1024)
        self.res_conv5 = nn.Conv2d(512, 1024, kernel_size=1)
        # --------------------------------------------------------------------------------------


        # Center Block (Switch Direction)
        self.center_conv1 = nn.Conv2d(1024, 512, 3, padding=4, dilation=4)  # Bigger Dilation Here
        self.center_gn1 = nn.BatchNorm2d(512)
        self.center_relu1 = nn.ReLU(inplace=True)

        self.center_conv2 = nn.Conv2d(512, 512, 3, padding=4, dilation=4)
        self.center_gn2 = nn.BatchNorm2d(512)
        self.center_relu2 = nn.ReLU(inplace=True)


        # -------------------------------  Decoder -----------------------------------------
        self.dec5 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=512)  # 512 + 512 -> 512
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)  # 512 + 256 -> 256
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)  # 256 + 128 -> 128
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)    # 128 + 64 -> 64
        # ------------------------------------------------------------------------------------------------


        # Final Segmentation Head: Predict n_class Channels
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0),  # Helps Generalise with Small Datasets

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0),  # Helps Generalise with Small Datasets

            nn.Conv2d(32, n_class, kernel_size=1)
        )


    def forward(self, x):
        # Stage 1
        residual = self.res_conv1(x)
        out = self.relu1_1(self.gn1_1(self.conv1_1(x)))
        out = self.relu1_2(self.gn1_2(self.conv1_2(out)))
        x = out + residual  # Residual Connection
        c1 = self.pool1(x)  # --> 1/2

        # Stage 2
        residual = self.res_conv2(c1)
        out = self.relu2_1(self.gn2_1(self.conv2_1(c1)))
        out = self.relu2_2(self.gn2_2(self.conv2_2(out)))
        x = out + residual  # Residual Connection
        c2 = self.pool2(x)  # --> 1/4

        # Stage 3
        residual = self.res_conv3(c2)
        out = self.relu3_1(self.gn3_1(self.conv3_1(c2)))
        out = self.relu3_2(self.gn3_2(self.conv3_2(out)))
        out = self.relu3_3(self.gn3_3(self.conv3_3(out)))
        x = out + residual  # Residual Connection
        c3 = self.pool3(x)  # --> 1/8

        # Stage 4
        residual = self.res_conv4(c3)
        out = self.drop4_1(self.relu4_1(self.gn4_1(self.conv4_1(c3))))
        out = self.drop4_2(self.relu4_2(self.gn4_2(self.conv4_2(out))))
        out = self.drop4_3(self.relu4_3(self.gn4_3(self.conv4_3(out))))
        x = out + residual  # Residual Connection
        c4 = self.pool4(x)  # --> 1/16

        # Stage 5
        residual = self.res_conv5(c4)
        out = self.drop5_1(self.relu5_1(self.gn5_1(self.conv5_1(c4))))
        out = self.drop5_2(self.relu5_2(self.gn5_2(self.conv5_2(out))))
        out = self.drop5_3(self.relu5_3(self.gn5_3(self.conv5_3(out))))
        x = out + residual  # Residual Connection

        # Center (Switch Direction)
        x = self.center_relu1(self.center_gn1(self.center_conv1(x)))
        x = self.center_relu2(self.center_gn2(self.center_conv2(x)))

        # Decoder with Concatenation
        x = self.dec5(x, c4)  # 1/16 scale
        x = self.dec4(x, c3)  # 1/8 scale
        x = self.dec3(x, c2)  # 1/4 scale
        x = self.dec2(x, c1)  # 1/2 scale

        # Final Upsample to Original Input Size (assumes input downscaled 1/2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Final Segmentation Prediction
        out = self.segmentation_head(x)  # (B, n_class, H, W)

        return out
    

    
# ---------------------- LOSS FUNCTION --------------------
class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, eps: float = 1e-7, beta: float = 1.0, reduction: str = "sum"):
        super(MulticlassDiceLoss, self).__init__()
        # Select Device: GPU if Available Else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a Tensor of Ones with Length = Num_classes, Used for Loss Calculation
        self.ones_tensor: torch.Tensor = torch.ones(num_classes, device=self.device)
        
        # Sum of Ones Tensor, Effectively the Number of Classes as a Float Tensor
        self.num_classes = torch.sum(self.ones_tensor)
        
        # Store Integer Number of Classes for Use in One-Hot Encoding & Masking
        self.num_classes_int = num_classes
        
        # Store the Reduction Method in Lowercase for Later Use
        self.reduction: str = reduction.lower()
        
        # Small Epsilon to Prevent Division by Zero in Calculations
        self.eps: float = eps
        
        # Square of Beta, Used in F-score Calculation to Weigh Precision Recall
        self.beta2: float = beta ** 2


    def multiclass_f_score(self, gt: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
        # Apply Softmax to Predictions to Get Class Probabilities Along Channel Dimension
        pr_softmax = torch.softmax(pr, dim=1)  # Shape: (B, C, H, W)

        # Replace Ignore Label (255) in Ground Truth With num_classes (to exclude in one-hot)
        gt[gt == 255] = self.num_classes_int

        # Convert Ground Truth to One-hot Encoding Including an Extra Class for Ignored Pixels
        gt_one_hot = torch.nn.functional.one_hot(gt, num_classes=self.num_classes_int + 1)  # (B, H, W, C+1)
        
        # Exclude the Ignored Class by Slicing
        gt_one_hot = gt_one_hot[:, :, :, :self.num_classes_int].float()
        
        # Change Shape from (B, H, W, C) to (B, C, H, W) to Align with Predictions
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2)  # (B, C, H, W)

        # True Positives: Sum Over Batch & Spatial Dims where Prediction & GT Agree
        tp = torch.sum(gt_one_hot * pr_softmax, dim=(0, 2, 3))
        
        # False Positives: Predicted Positives Minus True Positives
        fp = torch.sum(pr_softmax, dim=(0, 2, 3)) - tp
        
        # False Negatives: Actual Positives Minus True Positives
        fn = torch.sum(gt_one_hot, dim=(0, 2, 3)) - tp

        # Calculate F-score with Smoothing (eps) & Beta Weighting
        return ((1 + self.beta2) * tp + self.eps) / ((1 + self.beta2) * tp + self.beta2 * fn + fp + self.eps)


    def forward(self, pr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Calculate F-score for Each Class
        f_score = self.multiclass_f_score(pr=pr, gt=gt)
        
        # Dice Loss is 1 - F-score
        loss = self.ones_tensor.to(f_score.device) - f_score

        # Apply Specified Reduction to the Loss Tensor
        if self.reduction == "none":
            # No Reduction, Return Per-Class Loss
            pass
        elif self.reduction == "mean":
            # Mean Loss Over All Classes
            loss = loss.mean()
        elif self.reduction == "sum":
            # Sum Loss Over All Classes, Cast to float32 for Numerical Stability
            loss = loss.sum(dtype=torch.float32)
        elif self.reduction == "batchwise_mean":
            # Sum Over Batch Dimension & Average Per Class (if Loss is Batchwise)
            loss = loss.sum(dim=0, dtype=torch.float32)
            
        return loss



# ---------------------- TRAINING & EVALUATING --------------
def train_one_epoch(model, dataloader, criterion, optimizer, miou_metric):
    """Train the model for one full epoch."""
    model.train()              # Set Model to Training Mode
    running_loss = 0.0         # Initialise Cumulative Loss for the Epoch
    miou_metric.reset()        # Reset mIoU Metric Before Starting

    # Iterate Over Batches of Images & Labels
    for images, labels in tqdm(dataloader, desc="Training"):
        # Move Data to the Appropriate Device (CPU or GPU)
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()              # Clear Previous Gradients
        outputs = model(images)            # Forward Pass through the Model
        loss = criterion(outputs, labels)  # Compute Loss Between Predictions & Ground Truth
        loss.backward()                    # Backpropagate to Compute Gradients
        optimizer.step()                   # Update Model Weights

        running_loss += loss.item()     # Accumulate Loss for Averaging

        # Update mIoU Metric with Current Batch Predictions & Labels
        miou_metric.update(outputs.argmax(dim=1), labels)

    # Calculate Average Loss Over All Batches in the Epoch
    avg_loss = running_loss / len(dataloader)

    # Compute the Mean Intersection over Union Metric
    avg_miou = miou_metric.compute()

    return avg_loss, avg_miou


def evaluate(model, dataloader, miou_metric):
    """Evaluate the model on the validation/test set."""
    model.eval()             # Set Model to Evaluation Mode
    miou_metric.reset()      # Reset mIoU Metric

    # Disable Gradient Calculation for Evaluation
    with torch.no_grad():    
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            # Move Data to the Appropriate Device
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward Pass
            outputs = model(images)

            # Update mIoU Metric with Predictions & Labels
            miou_metric.update(outputs.argmax(dim=1), labels)

    # Compute Final mIoU After Processing All Batches
    return miou_metric.compute()



# ---------------------- MAIN TRAINING LOOP -------------------
def main():
    # Create Directory to Save the Model if it Doesn't Already Exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Prepare the Training Data Loader
    train_loader = get_data_loader(TRAIN_DIR, BATCH_SIZE, split="train")

    # Initialise the Segmentation Model & Move it to the Computing Device (CPU/GPU)
    model = BetterSegNetwork(n_class=NUM_CLASSES).to(DEVICE)

    # Define the Loss Function (Multiclass Dice Loss)
    criterion = MulticlassDiceLoss(num_classes=NUM_CLASSES)

    # Setup the Optimiser (AdamW) with Learning Rate & Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Learning Rate Scheduler: Cosine Annealing Adjusts LR Over Epochs
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Metric for Evaluation: Mean Intersection Over Union (mIoU)
    miou_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES).to(DEVICE)

    # TensorBoard SummaryWriter to Log Training Metrics
    writer = SummaryWriter(LOG_DIR)

    # Track the Best mIoU Score to Save the Best Model Checkpoint
    best_miou = 0.0

    # Loop Over the Specified Number of Epochs
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Train the Model for One Epoch & Get the Loss & mIoU
        train_loss, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, miou_metric)

        # Print Training Loss & mIoU for this Epoch
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")

        # Log Training Loss & mIoU Values to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('mIoU/train', train_miou, epoch)

        # If Current mIoU is Better than the Best Recorded, Save the Model Checkpoint
        if train_miou > best_miou:
            best_miou = train_miou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved Best Model with mIoU: {best_miou:.4f}")

        # Update Learning Rate According to the Scheduler
        scheduler.step()

    # After Training, Load the Best Saved Model for Evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Prepare the Test Data Loader
    test_loader = get_data_loader(TEST_DIR, BATCH_SIZE, split="test")

    # Evaluate the Model on the Test Set
    final_miou = evaluate(model, test_loader, miou_metric)
    print(f"\nFinal mIoU on test set: {final_miou:.4f}")

    # Close the TensorBoard Writer to Release Resources
    writer.close()



# Execute main() if this Script is Run Directly
if __name__ == "__main__":
    main()