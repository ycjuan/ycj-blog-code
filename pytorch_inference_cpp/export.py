import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleMLP()
model.eval()

dummy_input = torch.randn(1, 4)

# --- Approach 1: TorchScript ---
scripted = torch.jit.script(model)
scripted.save("model.pt")
print("Saved model.pt (TorchScript)")

# --- Approach 2 & 3: ONNX (used by both ONNX Runtime and TensorRT) ---
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)
print("Saved model.onnx (ONNX Runtime + TensorRT)")
