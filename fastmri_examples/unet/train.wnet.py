import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST  # Or your dataset
import torch.nn.functional as F

from fastmri.losses import CTLoss, ReconstructionLoss  # your loss classes
from fastmri.models import Wnet


def train(model, train_loader, optimizer, recon_loss_fn, device):
    model.train()
    running_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)

        optimizer.zero_grad()

        recon, seg = model(images)  # Ignore segmentation output

        loss = recon_loss_fn(recon, images)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, val_loader, recon_loss_fn, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)

            recon, _ = model(images)

            loss = recon_loss_fn(recon, images)
            running_loss += loss.item()

    return running_loss / len(val_loader)


def main():
    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-3
    num_classes = 5  # for segmentation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data preparation
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])

    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, losses
    model = Wnet(in_channels=1, out_channels=1, num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ct_loss_fn = CTLoss(reduction="mean")
    recon_loss_fn = ReconstructionLoss(method="l2", reduction="mean")

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        train_loss = train(model, train_loader, optimizer, recon_loss_fn, device)
        val_loss = validate(model, val_loader, recon_loss_fn, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the final model
    torch.save(model.state_dict(), "wnet_final.pth")
    print("Training complete. Model saved to wnet_final.pth.")


if __name__ == "__main__":
    main()
