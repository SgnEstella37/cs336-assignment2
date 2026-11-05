
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("2. fc1 output:", x.dtype)

        x = self.ln(x)
        print("3. LayerNorm output:", x.dtype)

        x = self.fc2(x)
        print("4. Predicted logits:", x.dtype)

        return x


def main():
    device = "cuda"

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA CUDA GPU.")

    # Create a model whose parameters are stored in FP32.
    model = ToyModel(in_features=128, out_features=5).to(device)

    # A batch of 4 input examples, each with 128 features.
    x = torch.randn(4, 128, device=device)

    # Integer class labels for cross-entropy loss.
    targets = torch.randint(0, 5, (4,), device=device)

    print("1. Model parameters:", next(model.parameters()).dtype)

    # Enable FP16 automatic mixed precision on CUDA.
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        print("5. Loss:", loss.dtype)

    # Compute gradients using automatic differentiation.
    loss.backward()
    print("6. Parameter gradients:", model.fc1.weight.grad.dtype)


if __name__ == "__main__":
    main()
