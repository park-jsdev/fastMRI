import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from torchvision.datasets import FashionMNIST  # Or your dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

from fastmri.losses import CTLoss, ReconstructionLoss  # your loss classes
from fastmri.models import Wnet


def train(model, train_loader, optimizer, recon_loss_fn, device, epoch=None):
    model.train()
    running_loss = 0.0

    os.makedirs("encoder_outputs", exist_ok=True)

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()
        decoded, seg_map = model(images)

        loss = recon_loss_fn(decoded, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # # Save only the first image of the first batch in the epoch
        # if batch_idx == 0 and epoch is not None:
        #     os.makedirs("encoder_outputs", exist_ok=True)
        #
        #     enc_img = encoded[0]  # First image in batch
        #
        #     # If encoded has shape [C, H, W], treat as feature map
        #     if enc_img.ndim == 3:
        #         grid = make_grid(enc_img.unsqueeze(1), normalize=True, scale_each=True)
        #         save_image(grid, f"encoder_outputs/epoch_{epoch:02d}.png")
        #     else:
        #         print("Unexpected encoder output shape:", enc_img.shape)

        # Save debug outputs for first batch
        if batch_idx == 0 and epoch is not None:
            os.makedirs(f"epoch_outputs/epoch_{epoch:02d}/encoder", exist_ok=True)
            os.makedirs(f"epoch_outputs/epoch_{epoch:02d}/images", exist_ok=True)

            # Save original input
            original = images[0, 0].detach().cpu()  # shape: [H, W]
            save_image(original.unsqueeze(0), f"epoch_outputs/epoch_{epoch:02d}/images/original.png", normalize=True)

            # Save reconstruction (assuming output is 1 channel now)
            reconstruction = decoded[0, 0].detach().cpu()
            save_image(reconstruction.unsqueeze(0), f"epoch_outputs/epoch_{epoch:02d}/images/reconstruction.png",
                       normalize=True)

            original_np = original.numpy()
            reconstruction_np = reconstruction.numpy()

            seg_mask = torch.argmax(seg_map[0], dim=0).cpu().numpy()  # shape: [H, W]

            # Plot and save the segmentation mask as an image
            plt.figure(figsize=(4, 4))
            plt.imshow(seg_mask, cmap='tab10')
            plt.axis('off')
            plt.title('Segmentation Mask')
            plt.savefig(f"epoch_outputs/epoch_{epoch:02d}/images/segmentation_mask.png")
            plt.close()

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
    num_epochs = 50
    learning_rate = 1e-3
    num_classes = 8  # for segmentation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data preparation
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])

    train_dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
    val_dataset =FashionMNIST(root="./data", train=False, download=True, transform=transform)

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
        train_loss = train(model, train_loader, optimizer, recon_loss_fn, device, epoch=epoch + 1)
        val_loss = validate(model, val_loader, recon_loss_fn, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the final model
    torch.save(model.state_dict(), "wnet_final.pth")
    print("Training complete. Model saved to wnet_final.pth.")


if __name__ == "__main__":
    main()
