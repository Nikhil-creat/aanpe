import re, numpy as np
from typing import TypedDict
from model import CLASSES
RECORDS = [
 "Spirit rover wheel slip in soft sand at Troy; right-front wheel stuck, drive halted in 2009.",
 "Curiosity wheel damage from sharp bedrock ridges on Aeolis Palus; route planning avoided Hazard terrain.",
 "Opportunity ripple sand traversal at Purgatory Dune caused heavy slip and entrapment in 2005.",
 "Curiosity Dingo Gap crossing: loose sand on a ripple required careful approach angle and low speed.",
 "Large boulder fields reduce clearance; prefer firm bedrock-soil corridors with low slip."]
class Store:
    def __init__(s, docs, dim=512):
        s.d, s.dim = docs, dim; s.m = np.stack([s.emb(t) for t in docs])
    def emb(s, t):
        v = np.zeros(s.dim)
        for w in re.findall(r"\w+", t.lower()): v[hash(w) % s.dim] += 1
        return v / (np.linalg.norm(v) or 1)
    def search(s, q, k=2): return [s.d[i] for i in np.argsort(-(s.m @ s.emb(q)))[:k]]
STORE = Store(RECORDS)
class S(TypedDict, total=False):
    counts: dict; conf: float; risk: bool; hits: list; advice: str; log: list

def perception(st, pred=None, probs=None):  # pred: HxW ints, ignores 255/4
    p = np.asarray(pred); v = p[p < 4]; n = max(v.size, 1)
    c = {k: float((v == i).sum()) / n for i, k in enumerate(CLASSES)}
    st["counts"] = c; st["conf"] = float(probs) if probs is not None else 0.0
    st.setdefault("log", []).append("Perception: " + ", ".join(f"{k} {x:.0%}" for k, x in c.items())); return st
def critic(st):
    c = st["counts"]; st["risk"] = c["Sand"] > 0.15 or c["Big Rock"] > 0.05
    if st["risk"]:
        st["hits"] = STORE.search("sand wheel slip entrapment boulder bedrock hazard")
        st["log"].append("Critic: hazard, RAG retrieval triggered")
    else: st["hits"] = []; st["log"].append("Critic: within limits")
    return st
def planner(st):
    c = st["counts"]
    if not st["risk"]: st["advice"] = "CLEAR: proceed on current heading."
    else:
        tgt = "Bedrock" if c["Bedrock"] >= c["Soil"] else "Soil"
        st["advice"] = f"ALERT: High risk of wheel entrapment. Re-route steering +15 degrees toward {tgt} zone.\nPrecedent: " + " | ".join(st["hits"])
    st["log"].append("Path: " + st["advice"].splitlines()[0]); return st

def run_crew(pred, conf=None):
    st = {"log": []}
    try:
        from langgraph.graph import StateGraph, END
        g = StateGraph(S); g.add_node("p", lambda s: perception(s, pred, conf)); g.add_node("c", critic); g.add_node("r", planner)
        g.set_entry_point("p"); g.add_edge("p", "c"); g.add_edge("c", "r"); g.add_edge("r", END)
        return g.compile().invoke(st)
    except ImportError:
        return planner(critic(perception(st, pred, conf)))
