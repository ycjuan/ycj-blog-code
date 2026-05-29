import torch
import torch.nn as nn


class MlpScorer(nn.Module):
    """
    2-tower MLP scorer (query + doc), matching the MlpScorerGPU architecture.

    Layer 1 uses separate weight matrices for query and doc. The query
    projection (W1_query * query + b1) is computed once and broadcast as a
    bias across all docs: h1 = ReLU(W1_doc * docs + query_proj).
    This avoids concatenating query into every doc row, matching the C++
    chained-BIAS-epilogue pipeline.

    Config:
        query_dim    (Dq): embedding dim of the query tower
        doc_dim      (Dd): embedding dim of the doc tower
        hidden_sizes [H1, H2, ...]: sizes of each hidden layer (>= 1 element)
        num_heads       : number of output scores per doc
    """

    def __init__(self, query_dim: int, doc_dim: int,
                 hidden_sizes: list[int], num_heads: int):
        super().__init__()
        assert len(hidden_sizes) >= 1

        # Layer 1: separate projections, bias lives on the query side only
        self.w1_query = nn.Linear(query_dim, hidden_sizes[0], bias=True)
        self.w1_doc   = nn.Linear(doc_dim,   hidden_sizes[0], bias=False)

        # Hidden layers 2..L
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)

        # Output layer: linear -> sigmoid
        self.output = nn.Linear(hidden_sizes[-1], num_heads)

    def forward(self, query: torch.Tensor, docs: torch.Tensor) -> torch.Tensor:
        """
        query : [Dq]      — single query embedding
        docs  : [N, Dd]   — N doc embeddings
        returns [N, num_heads] — scores in (0, 1)
        """
        query_proj = self.w1_query(query)        # [H1]  broadcast bias
        h = torch.relu(self.w1_doc(docs) + query_proj)  # [N, H1]
        h = self.hidden(h)                       # [N, H_last]
        return torch.sigmoid(self.output(h))     # [N, num_heads]


# ---------------------------------------------------------------------------
# Instantiate with example dims (random weights — no training)
# ---------------------------------------------------------------------------
QUERY_DIM    = 64
DOC_DIM      = 128
HIDDEN_SIZES = [256, 128]
NUM_HEADS    = 2

model = MlpScorer(QUERY_DIM, DOC_DIM, HIDDEN_SIZES, NUM_HEADS)
model.eval()

dummy_query = torch.randn(QUERY_DIM)
dummy_docs  = torch.randn(5, DOC_DIM)   # 5 docs

# Sanity check
with torch.no_grad():
    out = model(dummy_query, dummy_docs)
print(f"Output shape: {out.shape}")   # expect [5, 2]
print(f"Output:\n{out}")

# --- Approach 1: TorchScript ---
scripted = torch.jit.trace(model, (dummy_query, dummy_docs))
scripted.save("model.pt")
print("Saved model.pt (TorchScript)")

# --- Approach 2 & 3: ONNX (used by both ONNX Runtime and TensorRT) ---
torch.onnx.export(
    model,
    (dummy_query, dummy_docs),
    "model.onnx",
    input_names=["query", "docs"],
    output_names=["scores"],
    dynamic_axes={
        "docs":   {0: "num_docs"},
        "scores": {0: "num_docs"},
    },
    opset_version=17,
)
print("Saved model.onnx (ONNX Runtime + TensorRT)")

# --- Approach 4: Pure CUDA — dump weights as raw float32 binary files ---
# Each Linear weight is [out_dim, in_dim] row-major (PyTorch default).
# Biases are [out_dim].
import os
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)

def save_tensor(tensor: torch.Tensor, name: str):
    path = os.path.join(weights_dir, name)
    tensor.detach().cpu().float().numpy().tofile(path)
    print(f"  Saved {path}  shape={list(tensor.shape)}")

print("Saving weights for pure CUDA inference:")
save_tensor(model.w1_query.weight, "w1_query.bin")  # [H1, Dq]
save_tensor(model.w1_query.bias,   "b1.bin")         # [H1]
save_tensor(model.w1_doc.weight,   "w1_doc.bin")     # [H1, Dd]
for i, layer in enumerate(model.hidden):
    if isinstance(layer, torch.nn.Linear):
        idx = i // 2  # each hidden block is Linear + ReLU
        save_tensor(layer.weight, f"w_hidden_{idx}.bin")
        save_tensor(layer.bias,   f"b_hidden_{idx}.bin")
save_tensor(model.output.weight, "w_out.bin")        # [num_heads, H_last]
save_tensor(model.output.bias,   "b_out.bin")        # [num_heads]

# Also save config so the C++ side knows the dims
import json
config = {
    "query_dim":    QUERY_DIM,
    "doc_dim":      DOC_DIM,
    "hidden_sizes": HIDDEN_SIZES,
    "num_heads":    NUM_HEADS,
}
with open(os.path.join(weights_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved {weights_dir}/config.json")
