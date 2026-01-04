from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, random_split
from typing import Any
import sys
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn

BATCH_SIZE = 32
EPOCHS = 50

class Model(nn.Module):

    def __init__(self, in_channels=1, num_classes=3):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_data():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])


    train_ds = datasets.ImageFolder("dataset/train", transform=transform)
    val_ds   = datasets.ImageFolder("dataset/val", transform=transform)
    test_ds  = datasets.ImageFolder("dataset/test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)
    test_loader  = DataLoader(test_ds, batch_size=32)

    return train_loader, val_loader, test_loader

def show_random_predictions(model, test_loader, device, classes, num_samples=9) -> None:

    model.eval()
    
    # Get all test data
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Select random samples
    indices = torch.randperm(len(all_images))[:num_samples]
    sample_images = all_images[indices]
    sample_labels = all_labels[indices]
    
    # Get predictions
    with torch.no_grad():
        sample_images_device = sample_images.to(device)
        outputs = model(sample_images_device)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu()
    
    # Plot the samples
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle('Model Predictions on Random Test Samples', fontsize=8)
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:

            img = sample_images[idx].squeeze().cpu().numpy()
            img = img * 0.3081 + 0.1307
            img = np.clip(img, 0, 1) 
            
            true_label = sample_labels[idx].item()
            pred_label = predictions[idx].item()

            img_rgb = np.stack([img, img, img], axis=-1)

            if true_label == pred_label:

                img_rgb[:, :, 0] *= 0.5
                img_rgb[:, :, 2] *= 0.5
                img_rgb[:, :, 1] = np.clip(img_rgb[:, :, 1] * 1.2, 0, 1)
                color = 'green'
            else:

                img_rgb[:, :, 1] *= 0.5
                img_rgb[:, :, 2] *= 0.5
                img_rgb[:, :, 0] = np.clip(img_rgb[:, :, 0] * 1.2, 0, 1)
                color = 'red'
            
            ax.imshow(img_rgb)
            ax.set_title(f'True: {classes[true_label]} | Pred: {classes[pred_label]}', 
                        color=color, fontsize=6, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_classes():
    with open("people.txt") as file:
        content = file.readlines()
    content = [name.strip() for name in content]
    return content

def main() -> None:

    train_loader, val_loader, test_loader = load_data()
    classes = get_classes()

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = Model().to(device)

    model.train()

    sys.stdout.flush()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, EPOCHS):
        epoch_losses = []
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        #TRAINING the neural network
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            y_pred = model(batch_X)

            loss = criterion(y_pred, batch_y)

            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            train_loss += loss.item()
            epoch_losses.append(loss.item())
            _, predicted = torch.max(y_pred, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        avg_epoch_loss = np.mean(epoch_losses)
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        #VALIDATING the neural network
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        if epoch % 10 == 0:
            print(f"\nEpoch [{epoch}/{EPOCHS}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)

    print(f"\nFinal Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    show_random_predictions(model=model, test_loader=test_loader, device=device, classes=classes, num_samples=9)
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()