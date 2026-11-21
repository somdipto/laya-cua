import os, re, ast, json, time, random, collections, torch, numpy as np
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
N = int(os.environ.get("SS_N", "200")); BUDGET = float(os.environ.get("SS_BUDGET_S", str(4.5 * 3600))); TAG = os.environ.get("TAG", "ss")
torch.set_num_threads(os.cpu_count())
ds = load_dataset("rootsautomation/ScreenSpot", split="test")
idx = list(range(len(ds))); random.Random(3).shuffle(idx); idx = idx[:N]
print("ScreenSpot total", len(ds), "using", len(idx), flush=True)
mp = int(os.environ.get("SS_MAXPIX", "400")) * 28 * 28
try:
    proc = AutoProcessor.from_pretrained("showlab/ShowUI-2B", min_pixels=64 * 28 * 28, max_pixels=mp)
except Exception as e:
    print("ShowUI processor failed to load:", repr(e)[:300], "-> using the processor of its base model Qwen/Qwen2-VL-2B-Instruct (ShowUI-2B is fine-tuned from it)", flush=True)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=64 * 28 * 28, max_pixels=mp)
model = Qwen2VLForConditionalGeneration.from_pretrained("showlab/ShowUI-2B", torch_dtype=torch.float32).eval()
SYS = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
res = []; rng = random.Random(9); t0 = time.time(); bad = 0
for k, i in enumerate(idx):
    if time.time() - t0 > BUDGET: print("time budget reached", flush=True); break
    r = ds[i]; img = r["image"].convert("RGB"); x1, y1, x2, y2 = r["bbox"]
    msgs = [{"role": "user", "content": [{"type": "text", "text": SYS}, {"type": "image"}, {"type": "text", "text": r["instruction"]}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt")
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    s = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    try:
        p = ast.literal_eval(s.strip()); px, py = float(p[0]), float(p[1]); ok = True
    except Exception:
        px = py = None; ok = False; bad += 1
    hit = bool(ok and x1 <= px <= x2 and y1 <= py <= y2)
    cx, cy = 0.5, 0.5
    res.append({"i": i, "type": r["data_type"], "source": r["data_source"], "hit": hit, "parsed": ok, "center_hit": bool(x1 <= cx <= x2 and y1 <= cy <= y2), "rand_hit": bool(x1 <= rng.random() <= x2 and y1 <= rng.random() <= y2)})
    if (k + 1) % 5 == 0: print(f"{k+1}/{len(idx)} acc {np.mean([a['hit'] for a in res]):.3f} unparsed {bad} ({time.time()-t0:.0f}s) last={s[:40]!r}", flush=True)
def acc(f, key="hit"): 
    rs = [a for a in res if f(a)]; return {"n": len(rs), "acc": round(float(np.mean([a[key] for a in rs])), 4) if rs else None}
out = {"n": len(res), "max_pixels_k28sq": mp // 784, "unparsed": bad, "model_acc": acc(lambda a: True), "center_baseline": acc(lambda a: True, "center_hit"), "random_point_baseline": acc(lambda a: True, "rand_hit"),
       "by_type": {t: acc(lambda a, t=t: a["type"] == t) for t in ("text", "icon")}, "by_source": {s: acc(lambda a, s=s: a["source"] == s) for s in sorted({a["source"] for a in res})}, "seconds": round(time.time() - t0)}
print("RESULT", json.dumps(out), flush=True)
json.dump({"summary": out, "rows": res}, open(f"screenspot_results_{TAG}.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4.6))
nm = ["centre point", "random point", "ShowUI-2B"]; v = [out["center_baseline"]["acc"], out["random_point_baseline"]["acc"], out["model_acc"]["acc"]]
b = ax.bar(nm, v, color=["#999", "#bbb", "#4c78a8"])
for bb, vv in zip(b, v): ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.01, f"{vv:.3f}", ha="center")
ax.set_ylim(0, 1); ax.set_ylabel("click inside target box"); ax.set_title(f"ScreenSpot subset (n={out['n']}): click-in-box accuracy. Grounding model only; Laya is not involved.", fontsize=8)
fig.text(0.5, 0.01, "Single run on a random subset, CPU float32, reduced image resolution. Not comparable to published full-benchmark numbers.", ha="center", fontsize=7)
fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(f"screenspot_results_{TAG}.png", dpi=150)
print("SS DONE", flush=True)
