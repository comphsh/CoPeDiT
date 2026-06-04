# Git 操作记录 — CoPeDiT 实验适配

本文档记录所有 Git 命令及其解释，便于学习和复现。

---

## 1. 分支管理

### 创建实验分支
```bash
git checkout -b experiment/CoPeDiT_adapt
```
- `checkout -b <branch_name>`: 创建新分支并切换过去
- 分支命名: `experiment/{method_name}_adapt` — 表示对某个方法的实验性适配
- **解释**: 始终保持 main 分支干净，实验修改在独立分支上进行，便于回退和对比

### 查看当前分支
```bash
git branch
```
- 列出所有本地分支，`*` 标记当前分支

### 查看仓库状态
```bash
git status
```
- 显示已修改/未暂存/未跟踪的文件列表

---

## 2. 文件追踪与提交

### 添加文件到暂存区
```bash
git add data/ .gitignore
git add train_CoPeVAE_Brain_adapted.py train_MDiT3D_Brain_adapted.py
git add eval.py
git add run_train.sh run_eval.sh
```
- `git add <path>`: 将文件加入暂存区（staging area）
- 支持目录、单文件、glob 模式
- **解释**: 暂存区是 "下一次提交的预览"，让你选择性提交

### 提交变更
```bash
git commit -m "feat: add MONAI-based BraTS2020 data module

Replace the original hardcoded BraTS2021/IXI data loading with a unified
MONAI-based data module for BraTS2020..."
```
- `git commit -m "<message>"`: 创建一个版本快照
- Commit message 格式: `<type>: <short description>\n\n<detailed body>`
- **Types**: `feat` (新功能), `fix` (修复), `docs` (文档), `refactor` (重构)
- 每次提交对应一项完成的功能修改

### 查看提交历史
```bash
git log --oneline
```
- `--oneline`: 每个提交显示一行（hash + message）

---

## 3. 文件忽略

### .gitignore 文件
```
results/        # 实验结果不提交
model_save/     # 预训练模型不提交
__pycache__/    # Python 缓存
.venv/          # 虚拟环境
```
- **解释**: `.gitignore` 告诉 Git 忽略特定文件和目录
- `results/`, `model_save/`, `datasets/` 等不提交到版本控制

---

## 4. 其他常用命令

### 查看文件差异
```bash
git diff                    # 工作区 vs 暂存区
git diff --staged           # 暂存区 vs 最新提交
git diff HEAD~1             # 最新提交 vs 前一次提交
```

### 撤销操作
```bash
git checkout -- <file>        # 撤销工作区修改（未暂存）
git reset HEAD <file>         # 取消暂存
git reset --soft HEAD~1       # 撤销最近一次提交（保留修改）
```

### 暂存工作区修改
```bash
git stash                   # 暂存当前修改
git stash pop               # 恢复最近的暂存
```

---

## 5. 本次实验的 Git 工作流

```
main (原始代码)
  │
  └── experiment/CoPeDiT_adapt (本次实验分支)
        │
        ├── commit 1: 数据模块 (data/BraTS2020_data.py)
        ├── commit 2: 训练脚本 (train_*_adapted.py)
        ├── commit 3: 评估脚本 (eval.py)
        └── commit 4: 启动脚本 (run_*.sh)
```

**原则：**
1. 每次完成一项独立功能就 commit
2. Commit message 清晰描述修改内容
3. 所有修改在 `experiment/` 分支进行，不污染 main
4. `results/`, `model_save/`, `.pt` 等产物加入 `.gitignore`
