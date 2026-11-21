import sys, os, re, json, random, math, time, collections, glob, torch, numpy as np
import laya
ACTIONS = {"click":"click or tap a UI element","type":"type text into a focused field","select":"choose an option from a dropdown","key":"press a key or shortcut","scroll":"scroll the view","navigate":"open a URL or switch app or tab","move":"move the mouse without clicking (hover)","wait":"wait for the page or app to load","drag":"drag and drop or select by dragging","done":"the task is already complete, stop"}
Q = {"action_type":{"type":"choice","instructions":"Which action type should the agent take next?","criteria":ACTIONS}}
def page_text(html, n=1500):
    h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", h); t = re.sub(r"\s+", " ", t).strip()
    return "Web page text: " + t[:n]
OUT = "laya_m2w_target"
torch.set_num_threads(os.cpu_count())
info = json.load(open("train_info.json")); train_rows = [0] * info["train_rows"]; res = {"train_loss": info["train_loss"]}
# ---------------- pipeline eval: ShowUI-2B picks the target, Laya decides the operation ----------------
import ast, io, gc
from datasets import load_dataset, Image as HFImage
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
Image.MAX_IMAGE_PIXELS = None
NP = int(os.environ.get("PIPE_N", "100")); VW, VH = 1280, 720
def pj(x): return json.loads(x) if isinstance(x, str) else x
def bb(c):
    a = pj(c["attributes"]); x, y, w, h = [float(v) for v in str(a["bounding_box_rect"]).split(",")]
    return x, y, w, h
def tgt_from(c):
    if c is None: return "Target element: none."
    a = pj(c["attributes"]) or {}
    keep = [k + "=" + str(a[k])[:40] for k in ("type", "role", "name", "aria_label", "placeholder", "id", "class", "title", "value") if a.get(k)]
    return "Target element: <" + str(c.get("tag")) + "> " + " ".join(keep[:6]) + "."
ds = load_dataset("osunlp/Multimodal-Mind2Web", split="test_task", streaming=True).cast_column("screenshot", HFImage(decode=False))
sel = []; per = collections.Counter(); seen = 0
for r in ds:
    seen += 1
    if seen > 8000 or len(sel) >= NP: break
    try:
        op = pj(r["operation"])["op"].upper()
        if op not in ("CLICK", "TYPE", "SELECT") or not r["pos_candidates"]: continue
        pc = pj(r["pos_candidates"][0]); x, y, w, h = bb(pc)
        if w <= 2 or h <= 2 or x < 0 or y < 0 or x + w > VW or y + h > VH: continue
        if per[r["annotation_id"]] >= 3: continue
        m = re.match(r"\[(\w+)\]\s*(.*?)\s*->", r["target_action_reprs"]); name = (m.group(2) if m else "").strip()
        if not name: continue
        img = Image.open(io.BytesIO(r["screenshot"]["bytes"])).convert("RGB")
        if img.width < VW * 0.9: continue
        img = img.crop((0, 0, min(img.width, VW), min(img.height, VH)))
        cands = [pc] + [pj(c) for c in r["pos_candidates"][1:]] + [pj(c) for c in r["neg_candidates"]]
        boxes = []
        for c in cands:
            try: boxes.append((c, bb(c)))
            except Exception: pass
        idx = int(r["target_action_index"]); hist = []
        for z in r["action_reprs"][:idx]:
            mm = re.match(r"\[(\w+)\].*->\s*(\w+)", z)
            if mm: hist.append(mm.group(2).lower() + " on " + mm.group(1))
        per[r["annotation_id"]] += 1
        sel.append({"op": op.lower(), "name": name, "img": img, "w": img.width, "h": img.height, "pc": pc, "boxes": boxes, "task": r["confirmed_task"], "page": page_text(r["cleaned_html"], 900), "hist": hist[-4:], "bbox": (x, y, w, h)})
    except Exception as e:
        continue
print("pipeline steps selected", len(sel), "from", seen, "rows scanned; ops", collections.Counter(z["op"] for z in sel), flush=True)
laya_ag = laya.Agent(OUT, device="cpu")
def laya_op(z, c):
    st = {"task": z["task"], "screen": tgt_from(c) + " " + z["page"], "history": z["hist"]}
    return laya_ag.predict(st, Q)["answers"]["action_type"]["choice"]
proc = AutoProcessor.from_pretrained("showlab/ShowUI-2B")
sm = Qwen2VLForConditionalGeneration.from_pretrained("showlab/ShowUI-2B", torch_dtype=torch.float32).eval()
SYS = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
rows = []; t0 = time.time(); BUD = float(os.environ.get("PIPE_BUDGET_S", str(2.6 * 3600)))
for k, z in enumerate(sel):
    if time.time() - t0 > BUD: print("time budget reached", flush=True); break
    im = z["img"].resize((1008, 560), Image.BICUBIC)
    msgs = [{"role": "user", "content": [{"type": "text", "text": SYS}, {"type": "image"}, {"type": "text", "text": z["name"]}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[im], return_tensors="pt")
    with torch.no_grad(): out = sm.generate(**inp, max_new_tokens=32, do_sample=False)
    s_ = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    try:
        p = ast.literal_eval(s_.strip()); px, py = float(p[0]) * z["w"], float(p[1]) * z["h"]; ok = True
    except Exception: px = py = None; ok = False
    x, y, w, h = z["bbox"]; hit = bool(ok and x <= px <= x + w and y <= py <= y + h)
    chosen = None
    if ok:
        best = None
        for c, (cx, cy, cw, ch) in z["boxes"]:
            if cw > 2 and ch > 2 and cx <= px <= cx + cw and cy <= py <= cy + ch and (best is None or cw * ch < best[0]): best = (cw * ch, c)
        chosen = best[1] if best else None
    same = bool(chosen is not None and pj(chosen["attributes"]).get("backend_node_id") == pj(z["pc"]["attributes"]).get("backend_node_id"))
    o_or = laya_op(z, z["pc"]); o_sh = laya_op(z, chosen)
    rows.append({"gold": z["op"], "ground_hit": hit, "parsed": ok, "elem_match": same, "chosen_found": chosen is not None, "laya_oracle": o_or, "laya_showui": o_sh})
    if (k + 1) % 5 == 0: print(f"{k+1}/{len(sel)} ground {np.mean([a['ground_hit'] for a in rows]):.3f} oracle-op {np.mean([a['laya_oracle']==a['gold'] for a in rows]):.3f} showui-op {np.mean([a['laya_showui']==a['gold'] for a in rows]):.3f} ({time.time()-t0:.0f}s)", flush=True)
n = len(rows); g = [a["gold"] for a in rows]
def accf(key): return round(float(np.mean([a[key] == a["gold"] for a in rows])), 4)
def trec(key): return [sum(a[key] == "type" and a["gold"] == "type" for a in rows), g.count("type")]
summ = {"n": n, "gold_mix": collections.Counter(g).most_common(), "always_click": round(g.count("click") / n, 4), "laya_oracle_target_acc": accf("laya_oracle"), "laya_showui_target_acc": accf("laya_showui"),
        "type_recall_oracle": trec("laya_oracle"), "type_recall_showui": trec("laya_showui"), "showui_point_in_target_box": round(float(np.mean([a["ground_hit"] for a in rows])), 4),
        "showui_selected_exact_element": round(float(np.mean([a["elem_match"] for a in rows])), 4), "showui_point_on_any_candidate": round(float(np.mean([a["chosen_found"] for a in rows])), 4),
        "pred_mix_oracle": collections.Counter(a["laya_oracle"] for a in rows).most_common(), "pred_mix_showui": collections.Counter(a["laya_showui"] for a in rows).most_common(), "laya_train_rows": len(train_rows), "laya_train_loss": res["train_loss"]}
print("PIPE RESULT", json.dumps(summ), flush=True)
TAGP = os.environ.get("TAG", "pipe")
json.dump({"summary": summ, "rows": rows}, open(f"pipeline_results_{TAGP}.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.8))
nm = ["always-click", "Laya + oracle\ntarget", "Laya + ShowUI-2B\ntarget"]; v = [summ["always_click"], summ["laya_oracle_target_acc"], summ["laya_showui_target_acc"]]
b_ = ax.bar(nm, v, color=["#999", "#59a14f", "#4c78a8"])
for bb_, vv in zip(b_, v): ax.text(bb_.get_x() + bb_.get_width() / 2, vv + 0.01, f"{vv:.3f}", ha="center")
ax.set_ylim(0, 1.05); ax.set_ylabel("operation accuracy (click/type/select)")
ax.set_title(f"Small pipeline test, n={n} Mind2Web test_task steps. ShowUI point in true box: {summ['showui_point_in_target_box']:.2f}, exact element: {summ['showui_selected_exact_element']:.2f}", fontsize=8)
fig.text(0.5, 0.01, "Single run, small subset: only above-the-fold targets, ShowUI is given the element's name text. Not a full Mind2Web result.", ha="center", fontsize=7)
fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(f"pipeline_results_{TAGP}.png", dpi=150)
print("PIPE DONE", flush=True)
