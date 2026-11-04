import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from data_pipeline import IGNORE, N_CLS
CLASSES = ["Soil", "Bedrock", "Sand", "Big Rock"]

def build_model(encoder="tu-convnext_small", weights="imagenet"):
    return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=weights, classes=N_CLS)

class FocalDice(nn.Module):
    def __init__(self, gamma=1.5, eps=1e-6): super().__init__(); self.g, self.e = gamma, eps
    def forward(self, logits, y):
        v = (y != IGNORE).unsqueeze(1)
        p = F.softmax(logits, 1) * v
        oh = F.one_hot(y.clamp(max=N_CLS - 1), N_CLS).permute(0, 3, 1, 2).float() * v
        i, s = (p * oh).sum((0, 2, 3)), (p + oh).sum((0, 2, 3))
        return ((1 - (2 * i + self.e) / (s + self.e)) ** self.g).mean()

class CompositeLoss(nn.Module):
    """L = alpha * weighted CE (ignore 255) + beta * FocalDice (ignore 255)."""
    def __init__(self, weights, alpha=1.0, beta=1.0):
        super().__init__(); self.a, self.b = alpha, beta
        self.ce, self.fd = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE), FocalDice()
    def forward(self, logits, y): return self.a * self.ce(logits, y) + self.b * self.fd(logits, y)

def train(epochs=20, lr=3e-4, out="aanpe.pt"):
    from data_pipeline import loaders
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, _, w = loaders(); m = build_model().to(dev)
    crit, opt = CompositeLoss(w.to(dev)), torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        m.train(); tot = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(); l = crit(m(x), y); l.backward(); opt.step(); tot += l.item()
        sch.step(); print(f"epoch {ep + 1} loss {tot / len(tr):.4f}"); torch.save(m.state_dict(), out)

if __name__ == "__main__": train()
