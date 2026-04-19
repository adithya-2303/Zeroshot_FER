import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.emotion_cnn import EmotionCNN


# -----------------------------
# Settings
# -----------------------------

BATCH_SIZE = 64
EPOCHS = 25
LR = 0.001

MODEL_PATH = "zsl_model.pth"

UNSEEN_CLASS = "disgust"   # Zero-shot emotion


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
# Dataset
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


print("Training Classes:", train_data.classes)
print("Unseen Class:", UNSEEN_CLASS)


# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model
# -----------------------------

num_classes = len(train_data.classes)

model = EmotionCNN(num_classes=num_classes).to(device)


# -----------------------------
# Loss & Optimizer
# -----------------------------

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


# -----------------------------
# Training
# -----------------------------

print("\nStarting Zero-Shot Training...\n")

best_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
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
        _, pred = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()


    avg_loss = running_loss / len(train_loader)
    acc = 100 * correct / total


    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Acc: {acc:.2f}%"
    )


    # Save Best
    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(model.state_dict(), MODEL_PATH)

        print("✔ Best model saved")


print("\nTraining Finished")
print("Saved:", MODEL_PATH)