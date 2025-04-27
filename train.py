import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import JEPA
from dataset import create_wall_dataloader
import os


def jepa_loss(preds, targets):
    return F.mse_loss(preds, targets)


def train(model, dataloader, optimizer, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        valid_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            try:
                states = batch.states.to(device)
                actions = batch.actions.to(device)

                if actions.shape[1] == 0:
                    print(f"Skipping batch {batch_idx} (zero-length actions)")
                    continue

                preds = model(states, actions)
                with torch.no_grad():
                    targets = model.compute_target_embeddings(states[:, 1:])

                loss = jepa_loss(preds[:, 1:], targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                valid_batches += 1

                print(f"Batch {batch_idx} loss: {loss.item():.8f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                raise e

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            print(f"Epoch {epoch + 1} complete - average loss: {avg_loss:.8f}")
        else:
            print(f"Epoch {epoch + 1} complete - no valid batches")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"model_weights_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPA().to(device)

    train_loader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    total_epochs = 30  # you can change this as needed
    train(model, train_loader, optimizer, device, epochs=total_epochs)

    torch.save(model.state_dict(), "model_weights.pth")
    print("Final model saved to model_weights.pth")

