<div align="center">
  
  # AgentMark

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
- [Directory Structure](#-directory-structure)
- [Quick Start](#-quick-start)
  - [1. Environment Setup](#1-️-environment-setup-agentmark)
  - [2. Start Dashboard UI](#2-start-dashboard-ui)
  - [3. Dataset Setup](#3-dataset-setup)
  - [4. Configure Environment Variables](#4-configure-environment-variables)
- [One-run Flow (Plugin Mode)](#-one-run-flow-plugin-mode)
- [License](#license)
---

## 📂 Directory Structure

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

### 1. ⚙️ Environment Setup (AgentMark)

**For ToolBench and ALFWorld experiments (Python 3.9)**

Use Conda to manage the environment:

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate AgentMark

# Or install manually
pip install -r requirements.txt
```

### 2. Start Dashboard UI

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

### 3. Dataset Setup

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

### 4. Configure Environment Variables

Copy and edit the environment template:

```bash
cp .env.example .env
vim .env
# Fill in your API key (OpenAI / DeepSeek etc.)
# Use 'export KEY=VALUE' format or apply with:
export $(grep -v '^#' .env | xargs)
```

---

## ✅ One-run Flow (Plugin Mode)

This flow validates: **free-form user input → Swarm produces tools → proxy runs watermark sampling → Swarm executes tool_calls**.

### Step 1: Start AgentMark Proxy

```bash
cd /mnt/c/Users/25336/Desktop/AgentMarkWeb
source ~/miniconda3/etc/profile.d/conda.sh && conda activate AgentMark

export DEEPSEEK_API_KEY=sk-your-key
export TARGET_LLM_MODEL=deepseek-chat
export AGENTMARK_DEBUG=1
export AGENTMARK_TWO_PASS=0   # build tool_calls in proxy

uvicorn agentmark.proxy.server:app --host 0.0.0.0 --port 8001
```

### Step 2: Start Frontend (Visualization)

```bash
cd /mnt/c/Users/25336/Desktop/AgentMarkWeb/dashboard
npm install
npm run dev
```

Open: `http://localhost:5173`

### Step 3: Run Swarm (External Agent)

```bash
cd /mnt/c/Users/25336/Desktop/AgentMarkWeb/swarm
pip install -e .

export OPENAI_BASE_URL=http://localhost:8001/v1
export OPENAI_API_KEY=anything

python -m pytest -q examples/weather_agent/evals.py -k test_calls_weather_when_asked --disable-warnings -s
```

### Step 4: Verify Logs

In the **proxy terminal**, you should see:

- `[agentmark:scoring_request]`: scoring instruction injection
- `[agentmark:tool_calls_proxy]`: proxy-built tool_calls with args
- `[watermark]`: watermark result and visualization data

In the **frontend**, you can view the session and watermark distribution plots.

> Note: Swarm candidate tools come from `agent.functions`. User input is just message content. The proxy extracts candidates from `tools` and performs watermark sampling.

---

## License

This project is licensed under the [MIT License](LICENSE).
