"""
AuditorAI — PyTorch example
Demonstrates wrapping a PyTorch nn.Module with AuditorAI.
Run: python examples/pytorch_example.py
Requires: pip install auditorai[pytorch]
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("This example requires PyTorch. Install with: pip install torch")
    raise SystemExit(1)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from auditorai import AuditorSystem, wrap, run_full_evaluation


# ── Step 1: Define a simple 2-layer MLP ──────────────────────────
class SimpleMLP(nn.Module):
    """Two-layer MLP for binary classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Step 2: Load and prepare data ────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)


# ── Step 3: Train the PyTorch model ──────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleMLP(input_dim=X_train_s.shape[1], n_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Convert to tensors
X_train_t = torch.from_numpy(X_train_s.astype(np.float32)).to(device)
y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(device)

# Training loop — 20 epochs
print("Training PyTorch model...")
model.train()
for epoch in range(20):
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20, Loss: {loss.item():.4f}")


# ── Step 4: Wrap with AuditorAI ─────────────────────────────────
# The PyTorchAdapter automatically handles:
#   - Setting model to eval mode
#   - Disabling gradients for inference
#   - Converting logits to probabilities via softmax
#   - Batched inference to avoid OOM
adapter = wrap(model, adapter_type="pytorch", n_classes=2, device=device)


# ── Step 5: Create and train the auditor system ──────────────────
system = AuditorSystem(adapter)
system.train(X_val_s, y_val)  # Train auditor on VALIDATION data

# Auto-tune the suppression threshold
system.auto_tune(X_val_s, y_val)

# ── Step 6: Full evaluation ──────────────────────────────────────
run_full_evaluation(system, X_test_s, y_test, output_dir="outputs/pytorch")
print("\nPyTorch example complete! Check outputs/pytorch/ for plots.")
