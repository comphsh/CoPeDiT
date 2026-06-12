# CoPeDiT Rule Compliance — Change Log

> Spec: `FgC2F-UDiff__TCI2024/train_eval_pipe.md` Section 4 RULES

| Rule | File(s) | Change |
|------|---------|--------|
| **0** | `train_CoPeVAE_Brain_adapted.py:308-323`, `train_MDiT3D_Brain_adapted.py:311-326`, `eval.py:209-222` | `--data_root` / `--datalist_dir` defaults from `os.environ.get("DATA_ROOT")` / `os.environ.get("DATALIST_DIR")`. `--task_dir` defaults from `$COMPARE_ROOT/results/task_{ts}`. `PROJ_ROOT = os.environ.get("COMPARE_ROOT", ...)`. |
| **0** | `run_train.sh:26-30`, `run_eval.sh:29-33` | Global path block: `export COMPARE_ROOT=... DATA_ROOT=... DATALIST_DIR=...` |
| **1** | `train_CoPeVAE_Brain_adapted.py:92`, `train_MDiT3D_Brain_adapted.py:93` | Both train scripts call `read_train_val_datalist(...)` for train+val only. `test.list` is never loaded during training. |
| **2** | `eval.py:101` | `read_datalist()` loads `test.list` only for inference. Each sample processed individually (`batch_size=1`). |
| **3** | `eval.py:188-194` | `eval.py` only saves synthetic NIfTI images under `prediction/`. No metric computation inside eval. GT not saved. |
| **4** | `eval.py` (entire file) | No metric code in eval. User has external `syn_metric.py` / global eval script for metrics. |
| **5** | `data/BraTS2020_data.py:53-71`, `eval.py:162` | `MASK_LIST` = 14 masks; `mask_to_string(mask)` produces 4-bit binary `[flair,t1,t1ce,t2]`. `1`=available, `0`=missing. |
| **6** | `data/BraTS2020_data.py:48` | `MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]` — fixed order, used consistently across all scripts. |
| **7a** | `train_CoPeVAE_Brain_adapted.py:297-299`, `train_MDiT3D_Brain_adapted.py:300-302` | Train output: `models/` (checkpoints), `tensorboard/` (TF events), `logs/train.log` (logging). Flat under `results/task_{ts}/`. |
| **7b** | `eval.py:150-151`, `eval.py:141-144`, `eval.py:197` | Eval output: `prediction/{mask_str}/{patient}/{patient}_{mod}.nii.gz`. 3-level tqdm: mask outer → patient inner. Estimated/actual time printed. |
| **7c** | (external) | `prediction_metric_result/` — produced by user's global eval script, not by eval.py. |
| **8** | `train_CoPeVAE_Brain_adapted.py:55-72`, `train_MDiT3D_Brain_adapted.py:57-74` | `setup_logger()` writes to both console + `logs/train.log`. `--log_interval 10` controls per-step frequency. |
| **8** | `train_CoPeVAE_Brain_adapted.py:240-249`, `train_MDiT3D_Brain_adapted.py:253-262` | Per-step: `Epoch {cur}/{total} \| Step {in_epoch}/{per_epoch} [global {gs}/{total}] \| Loss: {val} \| LR: {lr}` |
| **8** | `train_CoPeVAE_Brain_adapted.py:259-263`, `train_MDiT3D_Brain_adapted.py:273-277` | Epoch-end: `Epoch [{cur}/{total}] \| Loss: {avg} \| LR: {lr} \| Time: {sec}s` |
| **9** | `train_CoPeVAE_Brain_adapted.py:75-87`, `train_MDiT3D_Brain_adapted.py:78-89` | `save_checkpoint()` writes `{epoch, global_step, model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch_loss, lr, args}`. |
| **9** | `train_CoPeVAE_Brain_adapted.py:213-216`, `train_MDiT3D_Brain_adapted.py:209-212` | `0.pt` saved before training starts (initial weights). |
| **9** | `train_CoPeVAE_Brain_adapted.py:262-266`, `train_MDiT3D_Brain_adapted.py:277-281` | `latest.pt` saved every epoch (overwrites). |
| **9** | `train_CoPeVAE_Brain_adapted.py:269-275`, `train_MDiT3D_Brain_adapted.py:284-290` | `checkpoint_epoch_{N}.pt` saved every `--ckpt_interval` epochs (milestone). |
| **9** | `train_CoPeVAE_Brain_adapted.py:308-313`, `train_MDiT3D_Brain_adapted.py:327-332` | `final_model.pt` saved at end of training. |
