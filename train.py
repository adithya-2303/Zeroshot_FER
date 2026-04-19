import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.emotion_cnn import EmotionCNN


# -----------------------------
# Settings
# -----------------------------

BATCH_SIZE = 64
INITIAL_EPOCHS = 20      # First training
EXTRA_EPOCHS = 10        # Additional training
LEARNING_RATE = 0.001

MODEL_PATH = "emotion_model.pth"
BEST_MODEL_PATH = "best_emotion_model.pth"

RESUME_TRAINING = True   # Set False for fresh training


# -----------------------------
# Transforms
# -----------------------------

transform = transforms.Compose([

    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))

])


# -----------------------------
# Dataset & Loader
# -----------------------------

train_data = torchvision.datasets.ImageFolder(
    root="data/fer2013/train",
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Model
# -----------------------------

model = EmotionCNN(num_classes=7).to(device)


# Resume training if enabled
if RESUME_TRAINING:

    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded saved model. Continuing training...")
    except:
        print("No saved model found. Training from scratch.")


# -----------------------------
# Loss & Optimizer
# -----------------------------

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training Function
# -----------------------------

def train_model(start_epoch, total_epochs, best_loss):

    for epoch in range(start_epoch, total_epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


            running_loss += loss.item()


            # Accuracy
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        # Average Loss
        avg_loss = running_loss / len(train_loader)

        # Accuracy
        accuracy = 100 * correct / total


        print(
            f"Epoch [{epoch+1}/{total_epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )


        # Save Best Model
        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print("✔️ Best model saved!")


        # Save Latest Model
        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


    return best_loss


# -----------------------------
# Main Training
# -----------------------------

print("\n===== Starting Training =====\n")

best_loss = float("inf")


# Phase 1: Initial Training
print("Phase 1: Initial Training")

best_loss = train_model(
    start_epoch=0,
    total_epochs=INITIAL_EPOCHS,
    best_loss=best_loss
)


# Phase 2: Extra Training
print("\nPhase 2: Additional Training")

best_loss = train_model(
    start_epoch=INITIAL_EPOCHS,
    total_epochs=INITIAL_EPOCHS + EXTRA_EPOCHS,
    best_loss=best_loss
)


print("\n===== Training Complete =====")

print("Final Model:", MODEL_PATH)
print("Best Model :", BEST_MODEL_PATH)