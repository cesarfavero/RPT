# RPT

README 语言版本：

- English: [`README.md`](/Users/cesarfavero/Documents/MyPru/RPT/README.md)
- Português: [`README.pt-BR.md`](/Users/cesarfavero/Documents/MyPru/RPT/README.pt-BR.md)
- 中文: `README.zh-CN.md`
- Русский: [`README.ru.md`](/Users/cesarfavero/Documents/MyPru/RPT/README.ru.md)

这是一个包含两条相关路线的研究与工程仓库：

1. `BitNet / sparsity / QAT-STE` 研究路线，它在早期为论文和方法论提供了基础；
2. `gemma4_theory/`，这是当前的实用路线：从现代稠密 Gemma 2B 模型出发，进行剪枝、MLP 路径三值化、功能恢复，并通过我们自己的方法和 Triton 运行时生成可用的 bundle。

## 意图

这个仓库是一个研究和工程项目。

它记录了哪些实验做过、哪些有效、哪些失败、以及当前哪些内容已经可用。它不是销售材料；其中的结论应当被理解为技术性结论，并且仅在这里实际验证过的 benchmark 和环境范围内成立。

## 当前最重要的内容

如果你想直接查看当前最实际的结果，请从 [`gemma4_theory/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/README.md) 开始。

在本地 100 题基准上，该路线已经验证：

- 原始 HF `google/gemma-2-2b-it`：`0.90`
- 我们改进后的稠密 checkpoint `student_pruned`：`0.92`
- 加入 recovered adapter 后的 packed ternary 路径：`1.00`
- 我们集成的 Triton packed runtime：`1.00`

这是目前仓库中最明确的结论：

- 与原始 Gemma 2B 基线相比，本地结果更好；
- 我们把整个 MLP 路径转换成了 packed ternary 表示；
- 我们在这个 packed 路径上恢复了功能；
- 我们生成了一个可用于 Linux/NVIDIA/Colab 的运行时 bundle。

## 从 Microsoft 指导到我们自己的栈

BitNet/Microsoft 的材料在项目早期非常有用，它为三值化思路、剪枝和部署纪律提供了概念上的指导。

但当前的 Gemma 路线并不是 Microsoft BitNet runtime 的复制品。这个仓库里的实际可用栈现在是我们自己的：

- 我们自己的 Gemma 2B 剪枝/恢复路径
- 我们自己的 MLP packed ternary 导出
- 我们自己的 packed-path recovery 过程
- 我们自己的 structural runtime 和 Triton runtime 路径
- 我们自己的 Linux/NVIDIA/Colab 生产 bundle

更准确的表述是：

- Microsoft/BitNet 是一个有价值的起点参考
- 当前真正可运行的方法和 runtime 是我们自己的

## 各路线当前状态

### 1. BitNet / 论文路线

仓库中仍然保留了早期以 BitNet 为核心的工作：

- 论文源文件：`paper/rpt_sparsity_paper.tex`
- 部署脚本：`deploy/`
- 历史 notebook 与日志

这部分对于研究历史仍然重要，但它已经不是仓库中唯一活跃的价值来源。

### 2. Gemma 路线

当前最实用的路径位于 `gemma4_theory/`。

已验证的关键里程碑：

- 原始 HF Gemma 2B 在本地套件上的参考分数：`0.90`
- 我们改进后的稠密 checkpoint `student_pruned`：`0.92`
- 完整 MLP ternary expansion 恢复后：`0.92`
- packed/materialized 恢复路径：`1.00`
- Triton runtime 集成：`1.00`
- T4 上生产 bundle 经过 prewarm 后的验证：
  - `avg_tokens_per_sec ~= 5.66`
  - `min_latency ~= 0.35s`
  - `max_latency ~= 1.02s`

这意味着项目已经不再只是“理论验证”。它已经在支持的环境中拥有一个可工作的 runtime bundle，并且当前实用路径不再依赖 Microsoft BitNet 工具链。

## 独立 Runtime 仓库

独立的 runtime 与 CLI 现在位于单独仓库中：

- [cesarfavero/rpt-runtime](https://github.com/cesarfavero/rpt-runtime)

主 RPT 仓库仍然是研究与验证的事实来源。

## 从哪里开始

### 如果你想看研究叙事

打开：

- [`gemma4_theory/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/README.md)

### 如果你想看 packed artifact 和 runtime

打开：

- [`gemma4_theory/deploy/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/deploy/README.md)
- [`gemma4_theory/runtime/packed_recovered_l78/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/runtime/packed_recovered_l78/README.md)

### 如果你已经下载了生产 zip

推荐流程：

```bash
unzip production_triton_packed_recovered_l78.zip
cd production_triton_packed_recovered_l78
python -m venv .venv
source .venv/bin/activate
pip install -r runtime/requirements.txt
python runtime/run_triton_inference.py --model-dir . --prompt "Qual a capital da França?"
```

更详细的说明见 runtime 和 deploy 的 README。

## 仓库结构

- `paper/`: 论文源文件和发布相关文件。
- `deploy/`: 旧的 BitNet 部署路线。
- `gemma4_theory/`: 当前从稠密模型到剪枝、三值化和 runtime 的主路线。
- `sessions/`: 历史实验日志。
- `results/`: 已跟踪的 JSON/CSV 实验输出。

## 可复现性

这个仓库的规则很简单：

- notebook 会保存带时间戳的结果文件；
- 不手工编辑实验结果；
- 每个有价值的阶段都应以 commit + push 收尾。

## 许可证

MIT License (`LICENSE`)。
