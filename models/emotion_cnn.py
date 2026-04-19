import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):

    def __init__(self, num_classes=7, embed_dim=384):

        super(EmotionCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Feature size = 128*6*6 = 4608
        self.feature_dim = 128 * 6 * 6


        # 🔹 Projection: 4608 → 384
        self.project = nn.Linear(self.feature_dim, embed_dim)


        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)


    # -----------------------------
    # Feature Extractor
    # -----------------------------
    def forward_features(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)   # 4608

        return x


    # -----------------------------
    # Projected Embedding (for ZSL)
    # -----------------------------
    def forward_embedding(self, x):

        x = self.forward_features(x)

        x = self.project(x)   # → 384

        return x


    # -----------------------------
    # Normal Classification
    # -----------------------------
    def forward(self, x):

        x = self.forward_features(x)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.fc2(x)

        return x