import os, cv2, numpy as np, torch
from fastapi import FastAPI, UploadFile, HTTPException
from model import build_model
from security_safeguards import guarded_infer
from agentic_rag import run_crew
from data_pipeline import MEAN, STD, SIZE

app = FastAPI(title="AANPE API", version="4.0")
M = build_model(weights=None).eval()
TRAINED = os.path.exists("aanpe.pt")
if TRAINED:
    M.load_state_dict(torch.load("aanpe.pt", map_location="cpu"))

@app.get("/health")
def health():
    return {"status": "ok", "trained_weights": TRAINED}

@app.post("/predict")
async def predict(file: UploadFile):
    raw = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), 1)
    if raw is None:
        raise HTTPException(400, "not an image")
    im = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
    x = torch.from_numpy(((im / 255. - MEAN) / STD).transpose(2, 0, 1)).float()[None]
    r = guarded_infer(M, x)
    if r["pred"] is None:
        return {"halt": True, "directive": r["directive"]}
    pred = r["pred"][0].numpy()
    st = run_crew(np.where(pred == 4, 255, pred), 1 - r["mean_entropy"] / np.log(4))
    return {"halt": r["halt"], "directive": r["directive"], "mean_entropy": r["mean_entropy"],
            "latency_ms": r["latency_ms"], "area": st["counts"], "advice": st["advice"],
            "trained_weights": TRAINED}
