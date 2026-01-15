<div align="center">
  
  <img src="assets/logo.svg" width="120" alt="AgentMark Logo" style="display: inline-block; vertical-align: middle; margin-right: 20px;"/>
  <img src="assets/logo-text.svg" width="300" alt="AgentMark" style="display: inline-block; vertical-align: middle;"/>
  
  **Behavioral Watermarking Framework for LLM Agents**

  [简体中文](README_zh.md) | [English](README.md)

  ![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
  ![License](https://img.shields.io/badge/license-MIT-green)
</div>

<div align="center">
  <img src="assets/overview.png" width="75%" alt="AgentMark Overview"/>
</div>

---


**AgentMark** is an experimental and evaluation framework for **behavioral watermarking of LLM agents**, implementing the utility-preserving and distribution-preserving watermark algorithms proposed in the **Agent Mark** paper.

The project provides a reproducible, modular, and extensible codebase to evaluate watermark performance, robustness, and stealth in complex agent tasks. It decomposes agent decision-making into **planning behavior** and **execution action**, embedding watermarks at the planning stage via distribution-preserving sampling to maintain downstream utility while enabling verifiable ownership protection.

<div align="center">
  <img src="assets/framework.png" width="100%" alt="AgentMark Framework"/>
</div>

### ✨ Key Features
- **💎 Utility Preservation**: Strict distribution-preserving sampling keeps watermarked behavior statistically indistinguishable from the original.
- **🛡️ Robustness**: Erasure-resilient coding and context-bound randomness handle missing logs and truncated trajectories.
- **🌍 Multi-environment Support**: Covers tool use, embodied intelligence, and social simulations.

### 🎮 Supported Environments
- **🛠️ ToolBench**: Complex tool-using scenarios with real-world API calls.
- **🏠 ALFWorld**: Text-based interactive household decision tasks.
- **📱 Oasis (Twitter/Reddit)**: Social-media behavior watermarking experiments.

---

## 📖 Table of Contents
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [1. Environment Setup](#1-️-environment-setup)
  - [2. Environment Variables](#2-environment-variables)
  - [3. Dataset Preparation](#3-dataset-preparation)
  - [4. Dashboard Visualization](#4-dashboard-visualization)
- [Using Our Plugin](#-using-our-plugin)
- [Experiment Guide](#-experiment-guide)
  - [1. ToolBench Tool Calling Experiment](#1-toolbench-tool-calling-experiment)
  - [2. ALFWorld Embodied Intelligence Experiment](#2-alfworld-embodied-intelligence-experiment)
  - [3. Oasis Social Media Experiment](#3-oasis-social-media-experiment)
  - [4. RLNC Robustness Evaluation](#4-rlnc-robustness-evaluation)
  - [5. Semantic Rewriting Robustness Evaluation](#5-semantic-rewriting-robustness-evaluation)
- [License](#license)
---

## 📂 Project Structure

```text
AgentMark/
├── assets/                         # Project assets (images, PDF)
├── agentmark/                      # Core library: watermark algorithms
│   ├── core/                       # Core watermark logic (ECC, sampling)
│   ├── environments/               # Environment adapters (ToolBench, ALFWorld)
│   └── data/                       # Bitstreams and configuration data
├── experiments/                    # Experimental implementations
│   ├── toolbench/                  # ToolBench API tool-use experiments
│   │   ├── scripts/                # Pipeline and analysis scripts
│   │   ├── configs/                # Pipeline config files
│   │   ├── tools/                  # Evaluation tools (StableToolBench)
│   │   ├── MarkLLM/                # SynthID watermark library (local mode)
│   ├── alfworld/                   # ALFWorld embodied intelligence experiments
│   │   ├── scripts/                # Experiment and analysis scripts
│   │   └── configs/                # Config files
│   ├── oasis_watermark/            # Social-media experiments
│   │   ├── twitter_watermark_experiment/  # Twitter simulation
│   │   ├── reddit_watermark_experiment/   # Reddit simulation
│   │   └── oasis/                  # Modified Oasis framework
│   ├── rlnc_trajectory/            # RLNC robustness evaluation
│   │   ├── scripts/                # Erasure eval and FPR analysis
│   │   └── *.json                  # Config files
│   └── semantic_rewriting/         # Semantic rewriting robustness tests
│       ├── scripts/                # Robustness test scripts
│       └── data/                   # Sample task data
├── output/                         # Logs, predictions, analysis outputs
├── environment.yml                 # Conda environment (Python 3.9)
├── requirements.txt                # Python dependencies (pip)
├── .env.example                    # Environment variable template
├── LICENSE                         # MIT License
└── README.md                       # English README
```

## 🚀 Quick Start

### 1. ⚙️ Environment Setup

**For ToolBench and ALFWorld experiments (Python 3.9)**

Use Conda to manage the environment:

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate AgentMark

# Or install manually
pip install -r requirements.txt
```

### 2. Environment Variables

Copy and edit the environment template:

```bash
cp .env.example .env
vim .env
# Fill in your API key (OpenAI / DeepSeek etc.)
# Use 'export KEY=VALUE' format or apply with:
export $(grep -v '^#' .env | xargs)
```

### 3. Dataset Preparation

#### ToolBench

> [!IMPORTANT]
> **ToolBench dataset is required!** You must complete the steps below before running ToolBench experiments.

**Download steps:**

1. **Download the ToolBench dataset**
   
   From the [ToolBench repository](https://github.com/OpenBMB/ToolBench), download the full dataset including:
   - `queries`: test query tasks
   - `tools`: API tool definitions (16,000+ tools)
   - `reference answers`: evaluation references

   ```bash
   # Recommended: use Git LFS or download from Releases
   # Dataset size ~2-3 GB
   ```

2. **Place into the correct directory**
   
   Put the extracted `data` folder under `experiments/toolbench/data/`:
   
   ```bash
   # Expected structure
   AgentMark/
   └── experiments/
       └── toolbench/
           └── data/
               └── data/           # extracted data folder
                   ├── test_query/
                   ├── toolenv/
                   │   └── tools/  # tool JSON definitions
                   └── answer/
   ```

3. **Verify dataset**
   
   Make sure `experiments/toolbench/data/data/toolenv/tools` contains multiple category subfolders (e.g., `Search/`, `Social_Media/`) and JSON tool files inside.

#### ALFWorld

The dataset is downloaded automatically to `~/.cache/alfworld`, or run manually:

```bash
alfworld-download
```

`experiments/alfworld/configs/base_config.yaml` is preconfigured to `/root/.cache/alfworld`.

> [!NOTE]
> Oasis (social media) experiments require a separate environment (Python 3.10+). Please refer to the [Oasis Social Media Experiments](#3-oasis-social-media-experiments) section below.

### 4. Dashboard Visualization

The dashboard provides interactive watermark experiments with real-time comparison and decoding analysis.

#### Requirements
- **Node.js**: 18.0+ (LTS recommended)
- **NPM**: comes with Node.js
- **Python**: backend runs in AgentMark environment

#### Steps

**Step 1: Start backend**

```bash
# Ensure you are in the project root
conda activate AgentMark
python dashboard/server/app.py
```

When you see `Uvicorn running on http://0.0.0.0:8000`, the backend is running.

> **Note**: backend listens on port **8000** by default.

**Step 2: Start frontend**

```bash
cd dashboard
npm install  # first time only
npm run dev
```

You will see a local URL, typically: `http://localhost:5173`

**Step 3: Open the app**

Visit `http://localhost:5173` or `http://127.0.0.1:5173` in your browser.

#### Common Issues

- **Port in use**: if 8000 or 5173 is occupied, stop the conflicting process or change config (frontend: `dashboard/vite.config.ts`, backend: `dashboard/server/app.py`).
- **Missing dependency**: if you see `ModuleNotFoundError`, install the missing package with `pip install <package>`.

---

## 🔌 Universal Plugin Integration

AgentMark provides a **Universal Proxy** that allows "one-click" watermark integration for any OpenAI-compatible agent framework (e.g., AutoGPT, LangChain, Swarm, LiteLLM) without modifying the agent's code.

**Core Concept**: Point your agent's `OPENAI_API_BASE_URL` to our proxy. The proxy intercepts requests, injects watermark logic during tool selection, and forwards to the upstream LLM.

### 1️⃣ Start the AgentMark Proxy

Open a terminal and run:

```bash
cd AgentMark
source ~/miniconda3/etc/profile.d/conda.sh && conda activate AgentMark

# Configuration
export DEEPSEEK_API_KEY=sk-your-key           # Your upstream API Key
export TARGET_LLM_MODEL=deepseek-chat         # Upstream model
export AGENTMARK_DEBUG=1

# Start the Proxy Server (Port 8001)
uvicorn agentmark.proxy.server:app --host 0.0.0.0 --port 8001
```

### 2️⃣ Start the Dashboard

Open a second terminal:

```bash
cd AgentMark
conda activate AgentMark
# Start Backend
python dashboard/server/app.py
```

Open a third terminal:

```bash
cd AgentMark/dashboard
npm run dev
# Dashboard available at http://localhost:5173
```

### 3️⃣ Connect Your Agent ("Add Agent" Mode)

1.  Open the Dashboard at `http://localhost:5173`.
2.  Select **Add Agent** on the welcome screen.
3.  **Compatibility Guide**:
    *   **✅ Perfect Match (One-Click)**: `AutoGPT`, `Swarm`, `OpenAI Agents SDK`, `LangChain`, `LlamaIndex`, `LiteLLM`.
    *   **❌ Not Supported**: Local weight models (llama.cpp), SaaS Web interfaces.
4.  **Integration**:
    *   Simply configure your agent environment to use the proxy address: `http://localhost:8001/v1`
    *   Or use the Dashboard's chat interface which automatically routes through the proxy for testing.

```bash
# Example: Using standard OpenAI SDK with Proxy
export OPENAI_BASE_URL=http://localhost:8001/v1
export OPENAI_API_KEY=any-dummy-key  # Proxy handles auth
python your_agent_script.py
```

### 4️⃣ Verify Watermark

In the **Proxy Terminal**, you will see logs indicating successful interception:
- `[agentmark:scoring_request]`: Proxy injecting scoring prompts.
- `[watermark]`: Watermark bits embedded in the decision.

In the **Dashboard**, you can see the real-time visualization of the watermark distribution.

---

## 📚 Experiment Guide

Detailed experimental guides are as follows:

### 1. ToolBench Tool Calling Experiment
- **Overview**: Simulates real-world API calling scenarios to evaluate watermark impact on tool usage and robustness.
- **Directory**: `experiments/toolbench/`
- **Two Running Modes**:
  | Mode | Config (`use_local_model`) | Description |
  |------|---------------------------|-------------|
  | **API Mode** | `false` (default) | Calls remote LLM APIs (e.g., DeepSeek, OpenAI), watermark embedded via behavioral sampling |
  | **Local Mode** | `true` | Loads local models (e.g., Llama-3), combines with SynthID text watermarking |
- **Run Pipeline**:
  ```bash
  conda activate AgentMark
  # Run full pipeline (baseline/watermark/evaluation)
  python experiments/toolbench/scripts/run_pipeline.py
  ```
- **Key Config**: `experiments/toolbench/configs/pipeline_config.json`
  - Switch mode: modify `common_config.use_local_model` to `true` or `false`
  - Local mode requires `local_model_path` pointing to model weights

### 2. ALFWorld Embodied Intelligence Experiment
- **Overview**: Text-based interactive household decision tasks, evaluating watermark impact on agent planning and execution.
- **Directory**: `experiments/alfworld/`
- **Environment Install**:
  ```bash
  pip install alfworld  # Install on top of AgentMark environment
  ```
- **Run Pipeline**:
  ```bash
  conda activate AgentMark
  # Run full pipeline (baseline/watermark/evaluation)
  python experiments/alfworld/scripts/run_experiment.py --config experiments/alfworld/configs/config.json
  ```
- **Key Config**: `experiments/alfworld/configs/config.json`

### 3. Oasis Social Media Experiment
> [!NOTE]
> 1. The `oasis/` directory is a **modified submodule** containing customized watermark logic.
> 2. Use a separate `oasis` environment (Python 3.10+).

- **Environment Install**:
  ```bash
  # 1. Create environment (Python 3.10+ recommended)
  conda create -n oasis python=3.10 -y
  conda activate oasis
  
  # 2. Install Oasis package
  pip install camel-oasis
  ```
  See [Oasis README](experiments/oasis_watermark/oasis/README.md) for details.

- **Overview**: Simulates user behavior and watermark injection on Twitter and Reddit.
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
  - **Note**: Simulates AI-related discussions in the `r/TechFuture` community.

### 4. RLNC Robustness Evaluation
- **Overview**: Tests RLNC (Random Linear Network Coding) watermark scheme recovery under packet loss/erasure scenarios.
- **Directory**: `experiments/rlnc_trajectory/`
- **Core Scripts**:
  | Script | Function |
  |--------|----------|
  | `scripts/rlnc_step_erasure_eval.py` | Erasure robustness evaluation (simulates various packet loss rates) |
  | `scripts/analyze_fpr.py` | **False Positive Rate (FPR) analysis** - simulates "no watermark" and "wrong key" attack scenarios |
- **Run Robustness Evaluation**:
  ```bash
  cd experiments/rlnc_trajectory
  python scripts/rlnc_step_erasure_eval.py --config rlnc_eval_config.json
  ```
- **Run FPR Analysis**:
  ```bash
  python scripts/analyze_fpr.py --config rlnc_fpr_config.json
  ```
- **Key Configs**: `rlnc_eval_config.json`, `rlnc_fpr_config.json`

### 5. Semantic Rewriting Robustness Evaluation
- **Overview**: Tests differential watermark robustness against semantic rewriting attacks.
- **Directory**: `experiments/semantic_rewriting/`
- **Run**:
  ```bash
  cd experiments/semantic_rewriting
  python scripts/robustness_test.py \
      --task data/001_task_0.json \
      --bits data/decoded_bits.json \
      --steps 5
  ```

---

## License

This project is licensed under the [MIT License](LICENSE).
