import torch
import torch.nn as nn


class BaseLinear(nn.Module):
    def __init__(self, dim_emb=512, n_classes=5):
        """
        Baseline Linear probing model.
        Takes an embedding as the input, outputs class-level predictions.
        Args:
            dim_emb (int, optional): Dimensionality of the embedding. Defaults to 512.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
        """
        super(BaseLinear, self).__init__()
        self.dim_emb = dim_emb
        self.n_classes = n_classes

        # A single linear layer will be used as the classifier
        self.classifier = nn.Linear(self.dim_emb, self.n_classes)

    def forward(self, emb):
        out = self.classifier(emb)
        return out

    def loss(self, out, target):
        cls_loss = nn.CrossEntropyLoss()(out, target)
        return cls_loss

    def trainable_params(self):
        return self.classifier.parameters()


