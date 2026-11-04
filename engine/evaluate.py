import os, numpy as np, torch, cv2
from data_pipeline import loaders, IGNORE, N_CLS
from model import build_model, CLASSES
def cm_update(cm, p, y):
    k = y != IGNORE; cm += np.bincount(N_CLS * y[k] + p[k], minlength=N_CLS ** 2).reshape(N_CLS, N_CLS)
def metrics(cm):
    iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
    return dict(pixel_acc=np.diag(cm).sum() / max(cm.sum(), 1), iou=dict(zip(CLASSES, iou)), miou=iou.mean())
def evaluate(ckpt="aanpe.pt", out="anomalies", max_save=50):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(weights=None).to(dev); m.load_state_dict(torch.load(ckpt, map_location=dev)); m.eval()
    _, tests, _ = loaders(); os.makedirs(out, exist_ok=True); saved = 0
    for name, dl in tests.items():
        cm, base = np.zeros((N_CLS, N_CLS), np.int64), np.zeros((N_CLS, N_CLS), np.int64)
        for x, y in dl:
            with torch.no_grad(): p = m(x.to(dev)).argmax(1).cpu().numpy()
            y = y.numpy()
            for i in range(len(y)):
                cm_update(cm, p[i], y[i]); cm_update(base, np.zeros_like(p[i]), y[i])  # baseline: all soil
                fn = (y[i] == 2) & np.isin(p[i], [0, 1])  # sand mistaken for soil/bedrock
                if fn.sum() > 200 and saved < max_save:
                    vis = np.zeros((*fn.shape, 3), np.uint8); vis[fn] = (0, 0, 255); vis[(y[i] == 2) & ~fn] = (0, 200, 0)
                    cv2.imwrite(f"{out}/{name}_{saved}_fn_sand.png", vis); saved += 1
        print(name, "model", metrics(cm), "\n   baseline", metrics(base), "\n", cm)
if __name__ == "__main__": evaluate()
