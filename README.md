# laya-cua


Notebook: [laya_cua_finetune.ipynb](laya_cua_finetune.ipynb) (outputs cleared; runs on a free Colab T4).






Fine-tune [Laya](https://huggingface.co/convaiinnovations/laya) (421M ModernBERT-based decision encoder) to pick the next **computer-use action type** and whether the task is **done**, from a text description of the screen.




Model: https://huggingface.co/sodan/laya-cua (Apache-2.0)




## What it does
Given a task, a text screen description and the last few steps, one forward pass returns:
- the next action type: click, type, key, scroll, move, wait, drag, done
- the probability that the task is already finished




## What it does not do
No text generation, no click coordinates, no screenshots. It is a fast decision layer to sit next to whatever produces the target and the typed text. Unsloth does not apply: Laya is an encoder with a typed head, not an autoregressive LLM.




## Results (held-out, n=334 rows, single run each, free Colab T4)
| version | change | action_type | task_done |
|---|---|---|---|
| v0 | 200 AgentNet tasks, 2 epochs | 0.7665 | 0.9581 |
| v1 | 400 tasks, 2 epochs | 0.8084 | 0.9611 |
| base Laya | none | 0.1856 | 0.7066 |
| always "click" | none | 0.7156 | not computed |




Limits: small test set, one run per version, about 94% of rows are "not done" so task_done is near a trivial baseline, trained on AgentNet text descriptions only. Details and the live results log are on the model card.




![Computer-use results](assets/laya_cua_results.svg)




## Function-calling routing (separate task)
Laya also decides call / no-call and which function on glaive-function-calling-v2. Test tool_call accuracy 0.9573 (base 0.3042, always no-call 0.648). Which-function is near ceiling for base Laya, so that gain is not meaningful. One run per library version, no confidence intervals. Weights (now inside the one model repo): https://huggingface.co/sodan/laya-cua/tree/main/function_calling. Details: [function_calling/RESULTS.md](function_calling/RESULTS.md).




![Function-calling results](assets/laya_fc_results.svg)




## Files
- `agentnet_to_laya.py`: converts OpenCUA AgentNet (MIT) trajectories into Laya rows.
- `function_calling/laya_fc.py`: the function-calling routing run.
- Training: the official `laya.train` RLCD loop on one T4 (single run recipe: 2 epochs, encoder LR 2.5e-5, head LR 1e-4, batch 64). A notebook will be added to this repo.


## Mind2Web CPU fine-tune rerun (negative result)

Run on a free GitHub Actions runner (CPU only, $0): https://github.com/somdipto/laya-cua/actions/runs/44786704167. Script: `m2w_cpu_finetune.py`. Raw numbers: `results/m2w_cpu_results.json`, log: `results/m2w_cpu_log.txt`.

Task: next action type on Mind2Web (osunlp/Mind2Web, CC-BY-4.0), text of the cleaned HTML only, whole tasks held out, 1634 eval steps.

| Model | Accuracy | Type recall |
|---|---|---|
| always-click | 0.8286 | n/a |
| laya-cua v1, zero-shot | 0.7809 | 2/211 |
| fine-tuned, 1 epoch, 900 train rows, CPU | 0.8286 | 0/211 |

The fine-tuned model predicts "click" for every step, so it matches always-click and adds nothing. Single run, one seed, 900 of 6141 train rows, 512-token inputs, one epoch, because of the free CPU limit. This does not show what a full-data GPU fine-tune would do. The chart is on the model card: https://huggingface.co/sodan/laya-cua
