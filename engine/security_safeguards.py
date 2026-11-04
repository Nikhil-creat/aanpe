import time, numpy as np, torch, torch.nn.functional as F
H_MAX, LAT_MAX_MS, HALT = 1.0, 150.0, "SAFE HALT: uncertain terrain marked Restricted Hazard."

def sanitize(x):
    """Return (ok, reason). x: float tensor [B,3,H,W], normalised."""
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 3: return False, "bad shape/channels"
    if not torch.isfinite(x).all(): return False, "non-finite values"
    if x.abs().max() > 10: return False, "values out of range"
    smooth = F.avg_pool2d(x, 5, 1, 2)
    if (x - smooth).abs().mean() > 0.6: return False, "high-frequency noise / possible perturbation"
    return True, "ok"

def entropy_map(logits):
    p = F.softmax(logits, 1); return -(p * torch.log(p.clamp_min(1e-8))).sum(1)  # [B,H,W]

def guarded_infer(model, x, restricted_label=4):
    ok, why = sanitize(x)
    if not ok: return dict(pred=None, entropy=None, halt=True, directive=f"{HALT} ({why})")
    t0 = time.perf_counter()
    with torch.no_grad(): lg = model(x)
    ent = entropy_map(lg); pred = lg.argmax(1); ms = (time.perf_counter() - t0) * 1e3
    mean_h = ent.mean().item(); halt = mean_h > H_MAX or ms > LAT_MAX_MS
    if halt: pred = torch.where(ent > H_MAX * 0.5, torch.full_like(pred, restricted_label), pred)
    return dict(pred=pred, entropy=ent, mean_entropy=mean_h, latency_ms=ms, halt=halt,
                directive=HALT if halt else "Proceed")
