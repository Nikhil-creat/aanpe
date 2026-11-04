import torch
from model import build_model
from data_pipeline import SIZE

def export(ckpt="aanpe.pt", out="aanpe.onnx"):
    m = build_model(weights=None).eval()
    try:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    except FileNotFoundError:
        print("No checkpoint found, exporting untrained weights.")
    torch.onnx.export(m, torch.randn(1, 3, SIZE, SIZE), out, input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    print("saved", out)

if __name__ == "__main__":
    export()
