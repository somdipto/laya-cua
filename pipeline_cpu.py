import sys, os, re, json, random, math, time, collections, glob, torch, numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import laya
from laya.agent import _fix_tokenizer_config
from laya.common import build_model, collate_items
from laya.calibrate import fit_temperature_map
from laya.train import items_from_rows, calibration_records, encode_item, rlcd_loss, save_checkpoint, sigma_at, split_calibration
ACTIONS = {"click":"click or tap a UI element","type":"type text into a focused field","select":"choose an option from a dropdown","key":"press a key or shortcut","scroll":"scroll the view","navigate":"open a URL or switch app or tab","move":"move the mouse without clicking (hover)","wait":"wait for the page or app to load","drag":"drag and drop or select by dragging","done":"the task is already complete, stop"}
Q = {"action_type":{"type":"choice","instructions":"Which action type should the agent take next?","criteria":ACTIONS}}
def target_text(a):
    # ORACLE target: the recorded ground-truth element for this step (tag + a few attributes). Not predicted by any model.
    if not a.get("pos_candidates"): return "Target element: none."
    c = a["pos_candidates"][0]
    try: at = json.loads(c.get("attributes") or "{}")
    except Exception: at = {}
    keep = [k + "=" + str(at[k])[:40] for k in ("type", "role", "name", "aria_label", "placeholder", "id", "class", "title", "value") if at.get(k)]
    return "Target element: <" + str(c.get("tag")) + "> " + " ".join(keep[:6]) + "."
def page_text(html, n=1500):
    h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", h); t = re.sub(r"\s+", " ", t).strip()
    return "Web page text: " + t[:n]
d = snapshot_download("osunlp/Mind2Web", repo_type="dataset", allow_patterns=["data/train/*"])
rows = []
for f in sorted(glob.glob(d + "/data/train/*.json")):
    for task in json.load(open(f)):
        hist = []
        for a in task["actions"]:
            op = a["operation"]["op"].lower()
            rows.append({"task_id": task["annotation_id"], "state": {"task": task["confirmed_task"], "screen": target_text(a) + " " + page_text(a["cleaned_html"], 900), "history": hist[-4:]}, "questions": Q, "expected": {"action_type": op}})
            hist.append(op + " on " + (a["pos_candidates"][0]["tag"] if a["pos_candidates"] else "element"))

N_TRAIN = int(os.environ.get("N_TRAIN", "1200")); EVAL_N = int(os.environ.get("EVAL_N", "0")); TRAIN_MAXLEN = int(os.environ.get("TRAIN_MAXLEN", "512"))
torch.set_num_threads(os.cpu_count())
ids = sorted({r["task_id"] for r in rows}); random.Random(11).shuffle(ids)
ev_ids = set(ids[:int(len(ids) * 0.2)])
train_rows = [r for r in rows if r["task_id"] not in ev_ids]; eval_rows = [r for r in rows if r["task_id"] in ev_ids]
assert not ({r["task_id"] for r in train_rows} & {r["task_id"] for r in eval_rows})
random.Random(5).shuffle(train_rows)
BAL = os.environ.get("BALANCED", "0") == "1"
if BAL:
    by = collections.defaultdict(list)
    for r in train_rows: by[r["expected"]["action_type"]].append(r)
    cap = {"type": N_TRAIN // 3, "select": N_TRAIN // 6}
    sel = []
    for k, v in by.items():
        if k in cap: sel += v[:cap[k]]
    n_non = sum(1 for r in sel)
    sel += by.get("click", [])[:max(0, N_TRAIN - n_non)]
    train_rows = sel; random.Random(6).shuffle(train_rows)
    print("balanced train mix", collections.Counter(r["expected"]["action_type"] for r in train_rows).most_common(), flush=True)
else:
    train_rows = train_rows[:N_TRAIN]
if EVAL_N: eval_rows = eval_rows[:EVAL_N]
print("M2W CPU run: train rows", len(train_rows), "eval rows", len(eval_rows), "(eval tasks held out whole)", flush=True)
model_dir = snapshot_download("sodan/laya-cua"); _fix_tokenizer_config(model_dir)
tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
cfg = dict(cfg, max_len=TRAIN_MAXLEN, head_max_len=256)
max_len, head_max_len = cfg["max_len"], cfg["head_max_len"]
items, skipped = items_from_rows(tok, train_rows, max_len, head_max_len)
print("items", len(items), "skipped", skipped, flush=True)
def ev(agent):
    p = [agent.predict(r["state"], Q)["answers"]["action_type"]["choice"] for r in eval_rows]
    g = [r["expected"]["action_type"] for r in eval_rows]
    return {"acc": round(float(np.mean([a == b for a, b in zip(p, g)])), 4), "pred_mix": collections.Counter(p).most_common(5), "type_hit": sum(a == b == "type" for a, b in zip(p, g)), "type_n": g.count("type"), "n": len(g)}
gold = [r["expected"]["action_type"] for r in eval_rows]
res = {"balanced": os.environ.get("BALANCED","0")=="1", "n_train_rows": len(train_rows), "n_eval_rows": len(eval_rows), "always_click": round(gold.count("click") / len(gold), 4), "train_maxlen": TRAIN_MAXLEN, "cpu_count": os.cpu_count()}
print("always-click", res["always_click"], flush=True)
t0 = time.time()
if os.environ.get("SKIP_ZS", "0") == "1":
    res["zero_shot"] = {"acc": 0.7809, "type_hit": 2, "type_n": 211, "n": 1634, "note": "from the earlier run on the same eval split (not recomputed)"}
else:
    v1 = laya.Agent(model_dir, device="cpu"); res["zero_shot"] = ev(v1); print("zero-shot", res["zero_shot"], f"({time.time()-t0:.0f}s)", flush=True); del v1
device = torch.device("cpu"); OUT = "laya_m2w_target"; EPOCHS = 1
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
model.to(device).train()
train_items, calib_items = split_calibration(items, calib_max=100, calib_frac=0.1, seed=20261119)
MB, GA, GS = 4, 4, 4
enc = [p for n, p in model.named_parameters() if "encoder." in n]; head = [p for n, p in model.named_parameters() if "encoder." not in n]
opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.01)
total = max(1, (len(train_items) // (MB * GA)) * EPOCHS)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
t0 = time.time(); random.Random(42).shuffle(train_items); sigma = sigma_at(0, EPOCHS, 0.4, 0.1)
opt.zero_grad(set_to_none=True); step = 0; tot = 0.0; nb = math.ceil(len(train_items) / MB)
for b in range(0, len(train_items), MB):
    chunk = train_items[b:b + MB]
    batch = collate_items([[encode_item(tok, it, max_len, head_max_len) for it in chunk]], tok.pad_token_id)
    logits, act = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
    loss = rlcd_loss(logits.float(), batch["target"], batch["marker_mask"], batch["qtype"], sigma, samples=GS, w_sph=0.75, w_rps=1.0)
    win = min(GA, nb - (step // GA) * GA)
    (loss / win).backward(); step += 1; tot += loss.item()
    if step % GA == 0 or b + MB >= len(train_items):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 10 == 0: print(f"step {step}/{nb} loss {tot/step:.4f} ({time.time()-t0:.0f}s)", flush=True)
res["train_seconds"] = round(time.time() - t0); res["train_loss"] = round(tot / max(1, step), 4)
model.eval()
fitted = fit_temperature_map(calibration_records(model, tok, calib_items, device, max_len, head_max_len, batch_size=8))
out_cfg = dict(cfg, max_len=1024, fine_tuned=True, model_name="laya-cua-m2w-cpu", temperature=fitted["temperature"]); out_cfg.pop("temperature_by_options", None)
if fitted["temperature_by_options"]: out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
save_checkpoint(model, tok, out_cfg, OUT); del model, opt

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
