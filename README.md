# RioRAG

**RioRAG** is a reinforcement learning-based framework for improving long-form Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start

### 🔧 Installation

Please install the following dependencies:

- `verl==0.2.0`
- `vllm==0.8.2`
- `ragchecker`
- `refchecker`
- `demjson`

### 🗂️ Data Preparation

```
cd train/verl/verl
python data_preprocess/checklist_rl.py
```

### 🏋️‍♀️ Training

```
cd train/verl
bash scripts/run_checklist.sh
```

### 🧪 Evaluation

Please refer to the code in the `eval/` directory for evaluation.
