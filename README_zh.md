<div align="center">
  <img src="assets/logo.svg" width="120" alt="AgentMark Logo" style="display:inline-block; vertical-align:middle; margin-right:20px"/>
  <img src="assets/logo-text.svg" height="80" alt="AgentMark" style="display:inline-block; vertical-align:middle"/>

  **LLM Agent 行为水印实验框架**

  [简体中文](README_zh.md) | [English](README.md)

  ![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
  ![License](https://img.shields.io/badge/license-MIT-green)
</div>

<div align="center">
  <img src="assets/overview.png" width="75%" alt="AgentMark Overview"/>
</div>

---


**AgentMark** 是一个专注于 **LLM Agent 行为水印（Behavioral Watermarking）** 的实验与评测框架，实现了 **Agent Mark** 论文中提出的效用保持（Utility Preservation）和分布保留（Distribution-Preserving）水印算法。

本项目旨在提供一套可复现、模块化且易于扩展的代码库，用于评估水印算法在复杂 Agent 任务中的性能、鲁棒性及隐蔽性。核心机制将 Agent 的决策过程分解为 **规划行为（Planning Behavior）** 和 **执行动作（Execution Action）**，通过在规划阶段进行分布保留采样来嵌入水印，从而在不影响下游任务效用（Utility）的前提下实现可验证的版权保护。

<div align="center">
  <img src="assets/framework.png" width="100%" alt="AgentMark Framework"/>
</div>

### ✨ 主要特性：
- **💎 效用保持 (Utility Preservation)**: 通过严格的分布保留采样，确保加水印后的 Agent 行为分布与原始分布统计不可区分。
- **🛡️ 高鲁棒性 (Robustness)**: 采用抗擦除编码（Erasure-Resilient Coding）和环境上下文绑定的随机性，能有效应对日志缺失（Missing Logs）和轨迹截断（Trajectory Truncation）。
- **🌍 多环境支持**: 覆盖工具使用、具身智能及社交模拟等多种场景。

### 🎮 支持的实验环境：
- **🛠️ ToolBench**: 模拟真实世界 API 调用的复杂工具使用场景。
- **🏠 ALFWorld**: 基于文本的交互式家庭环境决策任务。
- **📱 Oasis (Twitter/Reddit)**: 社交媒体环境下的行为水印实验。

---

## 📖 目录
- [目录结构](#-目录结构)
- [快速开始](#-快速开始)
  - [1. 环境配置](#1-️-环境配置-agentmark)
  - [2. 启动 Dashboard 可视化界面](#2-启动-dashboard-可视化界面)
  - [3. 数据集配置](#3-数据集配置)
  - [4. 配置环境变量](#4-配置环境变量)
- [SDK 使用](#-sdk-使用)
- [实验指南](#实验指南)
  - [1. ToolBench 实验](#1-toolbench-实验)
  - [2. ALFWorld 实验](#2-alfworld-实验)
  - [3. Oasis 社交媒体实验](#3-oasis-社交媒体实验)
  - [4. RLNC 轨迹鲁棒性实验](#4-rlnc-轨迹鲁棒性实验)
  - [5. 语义重写鲁棒性实验](#5-语义重写鲁棒性实验)
- [License](#license)

---

## 📂 目录结构

```text
AgentMark/
├── assets/                         # 项目资源 (图片, PDF)
├── agentmark/                      # 核心库：水印算法实现
│   ├── core/                       # 核心水印逻辑 (ECC, 采样)
│   ├── environments/               # 环境适配器 (ToolBench, ALFWorld)
│   └── data/                       # 比特流和配置数据
├── experiments/                    # 实验实现
│   ├── toolbench/                  # ToolBench API 工具调用实验
│   │   ├── scripts/                # 流水线和分析脚本
│   │   ├── configs/                # 流水线配置文件
│   │   ├── tools/                  # 评测工具 (StableToolBench)
│   │   ├── MarkLLM/                # SynthID 水印库 (本地模式)
│   ├── alfworld/                   # ALFWorld 具身智能实验
│   │   ├── scripts/                # 实验和分析脚本
│   │   └── configs/                # 配置文件
│   ├── oasis_watermark/            # 社交媒体实验
│   │   ├── twitter_watermark_experiment/  # Twitter 模拟
│   │   ├── reddit_watermark_experiment/   # Reddit 模拟
│   │   └── oasis/                  # 修改后的 Oasis 框架
│   ├── rlnc_trajectory/            # RLNC 鲁棒性评测
│   │   ├── scripts/                # 擦除评测和 FPR 分析
│   │   └── *.json                  # 配置文件
│   └── semantic_rewriting/         # 语义重写鲁棒性测试
│       ├── scripts/                # 鲁棒性测试脚本
│       └── data/                   # 示例任务数据
├── output/                     # 实验生成的日志、预测答案和分析结果
├── environment.yml                 # Conda 环境配置 (Python 3.9)
├── requirements.txt                # Python 依赖 (pip)
├── .env.example                    # 环境变量模板
├── LICENSE                         # MIT License
└── README.md                       # English README
```

## 🚀 快速开始

### 1. ⚙️ 环境配置 (AgentMark)

**适用于 ToolBench 和 ALFWorld 实验 (Python 3.9)**

建议使用 Conda 管理环境：

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate AgentMark

# 或者手动安装
pip install -r requirements.txt
```

### 2. 启动 Dashboard 可视化界面

Dashboard 提供了交互式的水印实验界面，包含实时对比、解码分析等功能。

#### 环境要求
- **Node.js**: 18.0 或更高版本（推荐使用 LTS）
- **NPM**: 通常随 Node.js 一起安装
- **Python**: 后端需要 AgentMark 环境

#### 启动步骤

**步骤 1: 启动后端服务**

打开一个终端窗口，运行：

```bash
# 确保在项目根目录
conda activate AgentMark
python dashboard/server/app.py
```

成功提示：当您看到 `Uvicorn running on http://0.0.0.0:8000` 时，说明后端已成功启动。

> **注意**: 后端服务默认监听 **8000** 端口。

**步骤 2: 启动前端界面**

打开另一个终端窗口，运行：

```bash
cd dashboard
npm install  # 仅首次需要
npm run dev
```

终端会显示访问地址，通常为：`http://localhost:5173`

**步骤 3: 访问应用**

打开浏览器，访问 `http://localhost:5173` 或 `http://127.0.0.1:5173` 即可使用 AgentMark Dashboard。

#### 常见问题

- **端口被占用**: 如果 8000 或 5173 端口被占用，请检查是否有其他服务正在运行，或修改配置文件（前端: `dashboard/vite.config.ts`，后端: `dashboard/server/app.py`）。
- **依赖缺失**: 如果启动后端时报错 `ModuleNotFoundError`，请使用 `pip install <缺少包名>` 安装。

### 3. 数据集配置

#### ToolBench

> [!IMPORTANT]
> **ToolBench 数据集是必需的！** 运行 ToolBench 实验前必须完成以下步骤，否则会因缺少工具定义和测试查询而无法运行。

**下载步骤：**

1. **下载 ToolBench 数据集**
   
   从 [ToolBench 官方仓库](https://github.com/OpenBMB/ToolBench) 下载完整数据集，包含：
   - `queries`: 测试查询任务
   - `tools`: 工具 API 定义 (约 16,000+ 个工具)
   - `reference answers`: 参考答案 (用于评测)

   ```bash
   # 推荐使用 Git LFS 或从 Release 页面直接下载
   # 数据集大小约 2-3 GB
   ```

2. **放置到正确目录**
   
   将解压后的 `data` 文件夹放入 `experiments/toolbench/data/` 目录下：
   
   ```bash
   # 预期的目录结构
   AgentMark/
   └── experiments/
       └── toolbench/
           └── data/
               └── data/           # 解压后的数据文件夹
                   ├── test_query/
                   ├── toolenv/
                   │   └── tools/  # 包含所有工具 JSON 定义
                   └── answer/
   ```

3. **验证数据集**
   
   确认 `experiments/toolbench/data/data/toolenv/tools` 目录下包含多个分类子目录（如 `Search/`, `Social_Media/` 等），每个分类下有工具的 JSON 文件。

#### ALFWorld
数据集在运行时会自动下载到 `~/.cache/alfworld`，或者您可以手动运行：
```bash
alfworld-download
```
`experiments/alfworld/configs/base_config.yaml` 中的配置已预设为指向 `/root/.cache/alfworld`。

> [!NOTE]
> Oasis (社交媒体) 实验需要独立的运行环境 (Python 3.10+)，请参考下方的 [Oasis 社交媒体实验](#3-oasis-社交媒体实验) 章节。

### 4. 配置环境变量

复制并修改环境变量模板：

```bash
cp .env.example .env
vim .env
# 填入您的 API Key (OpenAI / DeepSeek 等)
# 注意：请在 .env 中使用 'export KEY=VALUE' 语法，或运行以下命令使其生效：
export $(grep -v '^#' .env | xargs)
```

---

## 🔧 SDK 使用

AgentMark 提供了封装好的 SDK，便于其他 Agent 开发者快速集成行为水印，并为前端可视化提供结构化日志。

### 1. 主要接口

```python
from agentmark.sdk import AgentWatermarker

wm = AgentWatermarker(payload_text="team123", mock=False)

# 采样（嵌入水印）
result = wm.sample(
    probabilities={"Search": 0.5, "Reply": 0.3, "Finish": 0.2},
    context="task123||step1",          # 建议接入方自定义，需在日志里保存
    history=["last observation"],      # 备用：若 context 为空，使用 history 生成 key
)
print(result.action)                   # 选中的动作
print(result.distribution_diff)        # 给前端画概率对比的结构化数据

# 解码（验证水印）
bits = wm.decode(
    probabilities={"Search": 0.5, "Reply": 0.3, "Finish": 0.2},
    selected_action=result.action,
    context=result.context_used,
    round_num=result.round_num,
)
print(bits)
```

**返回对象 `WatermarkSampleResult`**：
- `action`: 本步被选中的动作
- `bits_embedded`: 本步嵌入的比特数
- `bit_index`: 当前累积指针（下次采样从这里继续）
- `payload_length`: 整个水印比特串长度
- `context_used`: 生成密钥的上下文（需在日志中保存，解码用）
- `round_num`: 使用的轮次编号（默认内部自增，亦可外部传入）
- `target_behaviors`: 编码期的"目标集合"（检测用）
- `distribution_diff`: 给前端的可视化数据（原始概率/水印后分布/目标标记）
- `is_mock`: 是否为 mock 模式（前端联调用）

### 2. 必备输入契约

- **候选动作 + 概率**：必须提供一个 `Dict[str, float]`，算法会归一化。若接入方只能拿到最终动作文本而没有候选概率，则无法使用此行为水印方案。
- **context_for_key**：建议格式如 `task_id||step_id||obs_hash`，务必随日志存储，用于解码和验水印。
- **轮次 round_num**：默认内部自增；若接入方已有自己的 step 序号，可通过 `round_num` 传入保持同步。

### 3. Mock 模式（前端联调）

初始化传入 `mock=True` 即可：`AgentWatermarker(..., mock=True)`。此模式返回伪造的 `distribution_diff`，方便前端先联调 UI，记得在展示层标注为 mock。

### 4. 日志建议字段

- `step_id` / `round_num`
- `context`（与编码一致）
- `probabilities`（行为名及概率）
- `selected_action`
- `target_behaviors`
- `bits_embedded` / `bit_index`
- `distribution_diff`（可选，前端展示用）

### 5. Prompt 驱动（黑盒 API）集成

当外部 LLM 只能通过 Prompt 返回自报概率时，可以使用 `agentmark.sdk.prompt_adapter` 辅助函数。

**Prompt 模板示例**：
```
你必须返回 JSON：
{
  "action_weights": {"Action1": 0.8, "Action2": 0.15, "Action3": 0.05},
  "action_args": {"Action1": {...}, "Action2": {...}, "Action3": {...}},
  "thought": "简要原因"
}
要求 action_weights 覆盖候选，值可不精确归一化，我们会归一化；不得输出 JSON 以外的文本。
```

**解析与采样代码**：
```python
from agentmark.sdk import AgentWatermarker
from agentmark.sdk.prompt_adapter import (
    choose_action_from_prompt_output,
    PromptWatermarkWrapper,
)

wm = AgentWatermarker(payload_text="team123")

# 方式1: 直接解析
selected, probs_used = choose_action_from_prompt_output(
    wm,
    raw_output=llm_response_text,
    fallback_actions=["Search", "Reply", "Finish"],
    context="task123||step1",
    history=["last observation"],
)

# 方式2: 使用包装器
wrapper = PromptWatermarkWrapper(wm)
system_prompt = base_system_prompt + "\n" + wrapper.get_instruction()
result = wrapper.process(
    raw_output=llm_response_text,
    fallback_actions=["Search", "Reply", "Finish"],
    context="task123||step1",
    history=["last observation"],
)
# result["action"] 供执行；result["frontend_data"] 直接给前端/日志
```

> **注意**：自报概率的可信度低于真实 logits，统计显著性可能受影响；解析失败时会回退为均分分布。

### 6. 网关模式（零代码改动）

如果不想修改 Agent 代码，可以部署水印网关。

**启动网关**：
```bash
export DEEPSEEK_API_KEY=sk-your-key
uvicorn agentmark.proxy.server:app --host 0.0.0.0 --port 8000
```

**可选环境变量**（推荐配置 `AGENTMARK_TWO_PASS`）：
```bash
export AGENTMARK_TWO_PASS=1                 # tools 场景下启用两阶段
export AGENTMARK_PAYLOAD_BITS=1101          # 固定水印 payload
export AGENTMARK_SESSION_DEFAULT=demo       # 默认会话 key
export AGENTMARK_PROB_TEMPERATURE=2.0       # 概率温度(>1 更平坦)
export AGENTMARK_FORCE_UNIFORM=1            # 强制均匀分布（演示用）
```

**Agent 端调用**（无需修改代码）：
```python
# 原代码
client = OpenAI(base_url="https://api.deepseek.com/v1")

# 改为
client = OpenAI(base_url="http://localhost:8000/v1")
```

或设置环境变量：
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=anything
```

**网关响应格式**：
```json
{
  "watermark": {
    "mode": "tools|system|extra_body|bootstrap",
    "candidates_used": ["候选1","候选2"],
    "probabilities_used": {"候选1":0.4, "候选2":0.6},
    "action": "候选2",
    "frontend_data": {...},
    "decoded_bits": "11",
    "context_used": "proxy||step1",
    "round_num": 0,
    "raw_llm_output": "原始 LLM 文本"
  }
}
```

**候选提取优先级**：
1. `tools/functions`（推荐，从工具定义自动提取）
2. `system` message 中的 agentmark 元数据
3. `extra_body.agentmark.candidates` / 顶层 `candidates`
4. 无候选则 bootstrap（显式标记，可靠性较低）

**自定义字段示例**：
```python
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    extra_body={
        "candidates": ["候选1","候选2"],
        "context": "task||step1",
        "agentmark": {
            "session_id": "your-session-id"  # 跨请求累积
        }
    }
)
print(resp.watermark)  # 包含水印信息
```

### 7. 真实 LLM 测试示例

**DeepSeek 集成测试**：
```bash
# 1. 激活环境
conda activate AgentMark
export DEEPSEEK_API_KEY=sk-your-key

# 2. 启动网关
uvicorn agentmark.proxy.server:app --host 0.0.0.0 --port 8000

# 3. 运行测试脚本
PYTHONPATH=. python3 tests/fake_agent_llm.py \
  --payload 1101 \
  --rounds 1 \
  --task "今天晚上吃什么？"
```

输出包含：
- `[raw LLM output]`: 模型原始 JSON 响应
- `frontend distribution diff`: 原始 vs 水印重组的分布
- `decoded bits`: 应匹配 payload 前缀

**前端柱状图验证流程**：
```bash
# 1. 启动 Dashboard 后端（端口 8000）
python dashboard/server/app.py

# 2. 启动网关（端口 8001）
export AGENTMARK_TWO_PASS=1
uvicorn agentmark.proxy.server:app --host 0.0.0.0 --port 8001

# 3. 生成前端场景
python tests/frontend_bar_demo.py \
  --proxy-base http://localhost:8001/v1 \
  --dashboard-base http://localhost:8000 \
  --rounds 5

# 4. 启动前端查看
cd dashboard && npm run dev
# 浏览器打开 http://localhost:5173
```

### 8. 打包与安装（pip 形态）

```bash
# 打包
pip install build
python -m build

# 安装
pip install dist/agentmark_sdk-0.1.0-py3-none-any.whl

# 使用
from agentmark.sdk import AgentWatermarker, PromptWatermarkWrapper
```

### 9. 依赖说明

封装内部复用了 `agentmark/core/watermark_sampler.py`，仍依赖 `torch`。若接入方环境较轻量，可在后续迭代提供纯 Python 版本或 HTTP 服务封装

---

## 实验指南

详细的实验运行指南如下：

### 1. ToolBench 实验
- **简介**: 模拟真实世界 API 调用场景，评估水印对工具使用能力和鲁棒性的影响。
- **目录**: `experiments/toolbench/`
- **两种运行模式**:
  | 模式 | 配置项 (`use_local_model`) | 说明 |
  |------|---------------------------|------|
  | **API 模式** | `false` (默认) | 调用远程 LLM API (如 DeepSeek, OpenAI)，水印通过行为采样嵌入 |
  | **本地模式** | `true` | 加载本地模型 (如 Llama-3)，结合 SynthID 文本水印算法 |
- **运行流水线**:
  ```bash
  conda activate AgentMark
  # 运行完整流水线 (包含 baseline/watermark/评测)
  python experiments/toolbench/scripts/run_pipeline.py
  ```
- **关键配置**: `experiments/toolbench/configs/pipeline_config.json`
  - 切换模式: 修改 `common_config.use_local_model` 为 `true` 或 `false`
  - 本地模式需额外配置 `local_model_path` 指向模型权重路径

### 2. ALFWorld 实验
- **简介**: 基于文本的交互式家庭环境决策任务，评估水印对 Agent 规划与执行能力的影响。
- **目录**: `experiments/alfworld/`
- **环境安装**:
  ```bash
  pip install alfworld  # 需在 AgentMark 环境基础上安装
  ```
- **运行流水线**:
  ```bash
  conda activate AgentMark
  # 运行完整流水线 (包含 baseline/watermark/评测)
  python experiments/alfworld/scripts/run_experiment.py --config experiments/alfworld/configs/config.json
  ```
- **关键配置**: `experiments/alfworld/configs/config.json`

### 3. Oasis 社交媒体实验
> [!NOTE]
> 1. 本目录下的 `oasis/` 是 **修改后的子依赖库** (Modified Submodule)，包含定制化的水印逻辑。
> 2. 请使用独立的 `oasis` (Python 3.10+) 环境运行。

- **环境安装**:
  ```bash
  # 1. 创建环境 (建议 Python 3.10+)
  conda create -n oasis python=3.10 -y
  conda activate oasis
  
  # 2. 安装 Oasis 包
  pip install camel-oasis
  ```
  详细说明请参考 [Oasis README](experiments/oasis_watermark/oasis/README.md)。

- **简介**: 模拟 Twitter 和 Reddit 上的用户行为与水印注入。
- **目录**: `experiments/oasis_watermark/`
- **Twitter 实验**:
  - 目录: `experiments/oasis_watermark/twitter_watermark_experiment/`
  - **运行**:
    ```bash
    cd experiments/oasis_watermark/twitter_watermark_experiment
    # 需配置 config.py 或设置环境变量 DEEPSEEK_API_KEY
    python run_experiment.py
    # 运行评测
    python evaluate_metrics_llm.py
    ```
- **Reddit 实验**:
  - 目录: `experiments/oasis_watermark/reddit_watermark_experiment/`
  - **运行**:
    ```bash
    cd experiments/oasis_watermark/reddit_watermark_experiment
    python run_experiment.py
    # 运行评测
    python evaluate_metrics_llm.py
    ```
  - **说明**: 模拟 `r/TechFuture` 社区中关于 AI 话题的讨论。

### 4. RLNC 轨迹鲁棒性实验
- **简介**: 测试基于 RLNC (Random Linear Network Coding) 的水印方案在丢包/擦除场景下的恢复能力。
- **目录**: `experiments/rlnc_trajectory/`
- **核心脚本**:
  | 脚本 | 功能 |
  |------|------|
  | `scripts/rlnc_step_erasure_eval.py` | 擦除鲁棒性评测 (模拟不同丢包率) |
  | `scripts/analyze_fpr.py` | **误报率 (FPR) 分析** - 模拟"未加水印"和"错误密钥"攻击场景 |
- **运行鲁棒性评测**:
  ```bash
  cd experiments/rlnc_trajectory
  python scripts/rlnc_step_erasure_eval.py --config rlnc_eval_config.json
  ```
- **运行 FPR 分析**:
  ```bash
  python scripts/analyze_fpr.py --config rlnc_fpr_config.json
  ```
- **关键配置**: `rlnc_eval_config.json`, `rlnc_fpr_config.json`

### 5. 语义重写鲁棒性实验
- **简介**: 测试差分水印在面对语义重写攻击 (Semantic Rewriting Attack) 时的鲁棒性。
- **目录**: `experiments/semantic_rewriting/`
- **运行**:
  ```bash
  cd experiments/semantic_rewriting
  python scripts/robustness_test.py \
      --task data/001_task_0.json \
      --bits data/decoded_bits.json \
      --steps 5
  ```

## License

This project is licensed under the [MIT License](LICENSE).


