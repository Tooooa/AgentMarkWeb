<div align="center">

  # AgentMark

  **Experimental Framework for LLM Agent Behavioral Watermarking**

  [简体中文](README_zh.md) | [English](README.md)

  ![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
  ![License](https://img.shields.io/badge/license-MIT-green)
</div>

<div align="center">
  <img src="assets/overview.png" width="75%" alt="AgentMark Overview"/>
</div>

---


**AgentMark** is an experimental framework focused on **LLM Agent Behavioral Watermarking**, implementing the Utility Preservation and Distribution-Preserving watermarking algorithms proposed in the **Agent Mark** paper.

This project aims to provide a reproducible, modular, and extensible codebase for evaluating the performance, robustness, and stealthiness of watermarking algorithms in complex agent tasks. The core mechanism decomposes the agent's decision-making process into **Planning Behavior** and **Execution Action**, embedding watermarks via distribution-preserving sampling during the planning phase. This ensures verifiable copyright protection without compromising downstream task utility.

<div align="center">
  <img src="assets/framework.png" width="100%" alt="AgentMark Framework"/>
</div>

### ✨ Key Features:
- **💎 Utility Preservation**: Ensures that the watermarked agent behavior distribution is statistically indistinguishable from the original distribution through strict distribution-preserving sampling.
- **🛡️ Robustness**: Utilizes Erasure-Resilient Coding and environment-context-bound randomness to effectively handle Missing Logs and Trajectory Truncation.
- **🌍 Multi-Environment Support**: Covers various scenarios including tool use, embodied agents, and social simulation.

### 🎮 Supported Environments:
- **🛠️ ToolBench**: Complex tool-use scenarios simulating real-world API calls.
- **🏠 ALFWorld**: Text-based interactive household decision-making tasks.
- **📱 Oasis (Twitter/Reddit)**: Behavior watermarking experiments in social media environments.

---

## 📖 Table of Contents
- [Directory Structure](#-directory-structure)
- [Quick Start](#-quick-start)
  - [1. Environment Setup](#1-️-environment-setup)
  - [2. Dataset Setup](#2-dataset-setup)
  - [3. Configuration](#3-configuration)
- [Experiment Guides](#-experiment-guides)
  - [1. ToolBench Experiments](#1-️-toolbench-experiments)
  - [2. ALFWorld Experiments](#2-alfworld-experiments)
  - [3. Oasis Social Media Experiments](#3-oasis-social-media-experiments)
  - [4. RLNC Trajectory Robustness](#4-️-rlnc-trajectory-robustness)
  - [5. Semantic Rewriting Robustness](#5-️-semantic-rewriting-robustness)
- [License](#license)

---

## 📂 Directory Structure

```text
AgentMark/
├── assets/                         # Project assets (images, pdfs)
├── agentmark/                      # Core library: Watermarking algorithms
│   ├── core/                       # Core watermark logic (ECC, sampling)
│   ├── environments/               # Environment adapters (ToolBench, ALFWorld)
│   └── data/                       # Bitstream and config data
├── experiments/                    # Experiment implementations
│   ├── toolbench/                  # ToolBench API tool-calling experiments
│   │   ├── scripts/                # Pipeline and analysis scripts
│   │   ├── configs/                # Pipeline configuration files
│   │   ├── tools/                  # Evaluation tools (StableToolBench)
│   │   └── MarkLLM/                # SynthID watermark library (local mode)
│   ├── alfworld/                   # ALFWorld embodied agent experiments
│   │   ├── scripts/                # Experiment and analysis scripts
│   │   └── configs/                # Configuration files
│   ├── oasis_watermark/            # Social media experiments
│   │   ├── twitter_watermark_experiment/  # Twitter simulation
│   │   ├── reddit_watermark_experiment/   # Reddit simulation
│   │   └── oasis/                  # Modified Oasis framework
│   ├── rlnc_trajectory/            # RLNC robustness evaluation
│   │   ├── scripts/                # Erasure eval and FPR analysis
│   │   └── *.json                  # Configuration files
│   └── semantic_rewriting/         # Semantic rewriting robustness tests
│       ├── scripts/                # Robustness test scripts
│       └── data/                   # Sample task data
├── output/                     # Generated experiment logs and answers
├── environment.yml                 # Conda environment (Python 3.9)
├── requirements.txt                # Python dependencies (pip)
├── .env.example                    # Environment variable template
├── LICENSE                         # MIT License
└── README_zh.md                    # 中文文档
```

## 🚀 Quick Start

### 1. ⚙️ Environment Setup

We recommend using Conda (Python 3.9+):

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate AgentMark

# Or install manually
pip install -r requirements.txt
```

### 2. Dataset Setup

#### ToolBench
1. Download the ToolBench data (queries, tools, and reference answers) from the [official repository](https://github.com/OpenBMB/ToolBench).
2. Place the unzipped `data` folder into `experiments/toolbench/data/`.
   - Expected path: `experiments/toolbench/data/data/toolenv/tools`, etc.

#### ALFWorld
The dataset will be downloaded automatically to `~/.cache/alfworld` when needed, or you can manually run:
```bash
alfworld-download
```
The configuration in `experiments/alfworld/configs/base_config.yaml` is pre-set to point to `/root/.cache/alfworld`.

> [!NOTE]
> Oasis (Social Media) experiments require a separate runtime environment (Python 3.10+). Please refer to the [Oasis Social Media Experiments](#3-oasis-social-media-experiments) section below.

### 3. Configuration

Copy and configure the environment variables:

```bash
cp .env.example .env
vim .env
# Enter your API Keys (OpenAI / DeepSeek, etc.)
# IMPORTANT: Use 'export KEY=VALUE' syntax in .env or run:
export $(grep -v '^#' .env | xargs)
```

## 🧪 Experiment Guides

Detailed running instructions for each experiment are provided below:

### 1. 🛠️ ToolBench Experiments
- **Introduction**: Simulates real-world API calling scenarios to evaluate the impact of watermarking on tool-use capabilities and robustness.
- **Directory**: `experiments/toolbench/`
- **Two Running Modes**:
  | Mode | Config (`use_local_model`) | Description |
  |------|---------------------------|------|
  | **API Mode** | `false` (Default) | Calls remote LLM APIs (e.g., DeepSeek, OpenAI). Watermarks are embedded via behavioral sampling. |
  | **Local Mode** | `true` | Loads a local model (e.g., Llama-3) and combines it with the SynthID text watermarking algorithm. |
- **Run Pipeline**:
  ```bash
  conda activate AgentMark
  # Run the full pipeline (includes baseline, watermarked, and evaluation)
  python experiments/toolbench/scripts/run_pipeline.py
  ```
- **Key Config**: `experiments/toolbench/configs/pipeline_config.json`
  - Toggle Mode: Change `common_config.use_local_model` to `true` or `false`.
  - Local mode requires `local_model_path` to point to your model weights.

### 2. 🏠 ALFWorld Experiments
- **Introduction**: Text-based interactive household decision-making tasks to evaluate the impact of watermarking on agent planning and execution.
- **Directory**: `experiments/alfworld/`
- **Environment Setup**:
  ```bash
  pip install alfworld  # Must be installed on top of the AgentMark environment
  ```
- **Run Pipeline**:
  ```bash
  conda activate AgentMark
  # Run the full pipeline
  python experiments/alfworld/scripts/run_experiment.py --config experiments/alfworld/configs/config.json
  ```
- **Key Config**: `experiments/alfworld/configs/config.json`

### 3. 📱 Oasis Social Media Experiments
> [!NOTE]
> 1. The `oasis/` directory here is a **modified submodule** containing customized watermarking logic.
> 2. Please use a separate `oasis` environment (Python 3.10+) as described below.

- **Environment Setup**:
  ```bash
  # 1. Create environment (Python 3.10+ recommended)
  conda create -n oasis python=3.10 -y
  conda activate oasis
  
  # 2. Install Oasis package
  pip install camel-oasis
  ```
  See [Oasis README](experiments/oasis_watermark/oasis/README.md) for details.

- **Introduction**: Simulates user behavior and watermark injection on Twitter and Reddit.
- **Directory**: `experiments/oasis_watermark/`
- **Twitter Experiment**:
  - Directory: `experiments/oasis_watermark/twitter_watermark_experiment/`
  - **Run**:
    ```bash
    cd experiments/oasis_watermark/twitter_watermark_experiment
    # Configure config.py or set DEEPSEEK_API_KEY environment variable
    python run_experiment.py
    # Run evaluation
    python evaluate_metrics_llm.py
    ```
- **Reddit Experiment**:
  - Directory: `experiments/oasis_watermark/reddit_watermark_experiment/`
  - **Run**:
    ```bash
    cd experiments/oasis_watermark/reddit_watermark_experiment
    python run_experiment.py
    # Run evaluation
    python evaluate_metrics_llm.py
    ```
  - **Description**: Simulates discussions about AI topics in the `r/TechFuture` community.

### 4. 🛡️ RLNC Trajectory Robustness
- **Introduction**: Tests the recovery capability of the RLNC (Random Linear Network Coding) watermark scheme under log loss/erasure scenarios.
- **Directory**: `experiments/rlnc_trajectory/`
- **Core Scripts**:
  | Script | Function |
  |------|------|
  | `scripts/rlnc_step_erasure_eval.py` | Erasure robustness evaluation (simulates different loss rates). |
  | `scripts/analyze_fpr.py` | **False Positive Rate (FPR) Analysis** - Simulates non-watermarked and wrong-key attack scenarios. |
- **Run Robustness Eval**:
  ```bash
  cd experiments/rlnc_trajectory
  python scripts/rlnc_step_erasure_eval.py --config rlnc_eval_config.json
  ```
- **Run FPR Analysis**:
  ```bash
  python scripts/analyze_fpr.py --config rlnc_fpr_config.json
  ```
- **Key Configs**: `rlnc_eval_config.json`, `rlnc_fpr_config.json`

### 5. ✍️ Semantic Rewriting Robustness
- **Introduction**: Tests the robustness of differential watermarking against Semantic Rewriting Attacks.
- **Directory**: `experiments/semantic_rewriting/`
- **Run**:
  ```bash
  cd experiments/semantic_rewriting
  python scripts/robustness_test.py \
      --task data/001_task_0.json \
      --bits data/decoded_bits.json \
      --steps 5
  ```

## License

This project is licensed under the [MIT License](LICENSE).


