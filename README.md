# TravelBench: A Multi-Constraint Evaluation Benchmark for Travel Agents

This repository contains the official dataset, evaluation code, and baseline agent implementation for the paper **"TravelBench: Benchmarking LLM Agents on Comprehensive Travel Planning"**.

## 📂 Repository Structure

```
├── dataset/                # TravelBench queries and config
│   ├── travelbench_queries.csv
│   └── config/             # Facility mapping configs
├── api/                    # Travel Service Sandbox (Mock API)
│   ├── app.py
│   └── data/               # (Download from Google Drive)
├── agent/                  # Baseline ReAct Agent
│   └── run_agent.py
├── evaluation/             # Scoring System
│   ├── scoring.py
│   ├── implicit_scoring.py
│   └── data_loader.py
└── outputs/                # Evaluation results
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+

### 2. Setup Data

The full API database (215K records) is too large for git. Please download it from Google Drive and extract to `api/data/`:

👉 **[Download API Data](https://drive.google.com/drive/folders/1m-AdUqvZrkUMXkT8TU6Ryb58ruTxcWj4?usp=sharing)**

**Important:** You must extract the contents into the `api/data/` folder so the API server can find them. The structure should look like this:

```
api/data/
  ├── flights/            # Flight data csvs
  ├── hotels/             # Hotel data csvs
  ├── attractions/        # Attraction data csvs
  ├── cars/               # Car rental data csvs
  └── ...
```

### 3. Start the Travel API Sandbox

You must start the API server before running the agent.

```bash
cd api
pip install -r requirements.txt
python app.py
```
*Server runs at http://localhost:5000*

### 4. Configure & Run Agent

Set up your LLM credentials:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Run the baseline agent:
```bash
# Run on the first 10 queries
python agent/run_agent.py \
    --input dataset/travelbench_queries.csv \
    --output outputs/results.json \
    --limit 10
```

### 5. Evaluation

Score the generated plans:

```bash
python evaluation/scoring.py \
    --input outputs/results.json \
    --meta dataset/travelbench_queries.csv
```

## 📜 Citation

If you use TravelBench in your research, please cite:



## 📄 License

MIT License
