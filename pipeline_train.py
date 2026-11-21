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


json.dump({"train_rows": len(train_rows), "train_loss": res["train_loss"]}, open("train_info.json", "w"))
print("TRAIN PHASE DONE", res["train_loss"], flush=True)
