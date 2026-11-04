import os, numpy as np, torch, gradio as gr, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import build_model, CLASSES
from security_safeguards import guarded_infer
from agentic_rag import run_crew
from data_pipeline import MEAN, STD, SIZE
import cv2
COL = np.array([[60, 190, 90], [60, 110, 255], [255, 170, 40], [150, 90, 255], [255, 40, 40]], np.uint8)
M = build_model(weights=None).eval()
if os.path.exists("aanpe.pt"): M.load_state_dict(torch.load("aanpe.pt", map_location="cpu")); NOTE = ""
else: NOTE = "WARNING: no aanpe.pt found, predictions come from untrained weights.\n"
def analyse(img):
    if img is None: return None, None, "Upload an image.", None
    im = cv2.resize(img, (SIZE, SIZE)); x = torch.from_numpy(((im / 255. - MEAN) / STD).transpose(2, 0, 1)).float()[None]
    r = guarded_infer(M, x)
    if r["pred"] is None: return im, None, r["directive"], None
    pred = r["pred"][0].numpy(); ov = cv2.addWeighted(im, .5, COL[pred], .5, 0)
    st = run_crew(np.where(pred == 4, 255, pred), 1 - r["mean_entropy"] / np.log(4))
    fig, ax = plt.subplots(figsize=(4, 3)); ax.bar(CLASSES, [st["counts"][k] * 100 for k in CLASSES], color=COL[:4] / 255); ax.set_ylabel("% area")
    e = r["entropy"][0].numpy(); eh = cv2.applyColorMap((255 * e / np.log(4)).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    log = NOTE + "\n".join(st["log"]) + f"\nEntropy {r['mean_entropy']:.3f} | latency {r['latency_ms']:.0f} ms\n{r['directive']}\n\n{st['advice']}"
    return ov, fig, log, eh
with gr.Blocks(title="AANPE v4.0") as demo:
    gr.Markdown("# AANPE v4.0 Rover Terrain Safety"); inp = gr.Image(label="NavCam image", type="numpy")
    b = gr.Button("Analyse"); o1, o2, o4 = gr.Image(label="Segmentation"), gr.Plot(label="Area"), gr.Image(label="Entropy")
    o3 = gr.Textbox(label="Agentic safety log", lines=10); b.click(analyse, inp, [o1, o2, o3, o4])
if __name__ == "__main__": demo.launch(server_name="0.0.0.0", server_port=7860)
