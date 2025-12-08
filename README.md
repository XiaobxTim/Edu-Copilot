# EduCopilot
EduCopilot is an innovative AI education platform that delivers personalized learning experiences to students through a multi-agent system. It dynamically adjusts the difficulty of instruction based on student feedback, generates customized practice exercises, and provides real-time evaluation and feedback.

## Table of Content
- [EduCopilot](#educopilot)
  - [Table of Content](#table-of-content)
  - [Introduction](#introduction)
    - [Core Feature](#core-feature)
    - [System Architecture​](#system-architecture)
  - [Quick Start](#quick-start)
  - [Structure](#structure)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Introduction
### Core Feature
### System Architecture​
## Quick Start
**Environment Setup**
```
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

**Check Environment**
```
python check_env.py
```

**Step 1: Data Synthesis**
```
chmod +x ./scripts/run_sft_generator.sh
chmod +x ./scripts/run_preference_generator.sh
chmod +x ./scripts/run_data_filter.sh

# sft data generator
export DEEPSEEK_API_KEY="you api key"
./scripts/run_sft_generator.sh (--overwrite) # depend on whether overwriting exist data

# preference generator
./scripts/run_preference_generator.sh (--overwrite) # depend on whether overwriting exist data

# data filter
export HF_ENDPOINT=https://hf-mirror.com
git config --global credential.helper store
huggingface-cli login --token XXX(your access token)
./scripts/run_data_filter.sh
```

## Structure

```plaintext
edu-copilot/
|-- README.md
|-- check_env.py
|-- requirements.txt
|-- configs/
|-- data/
| |-- edu_copilot_sft_data.json
| |-- edu_copilot_sft_data.jsonl
| |-- edu_copilot_preference_data.json
| |-- edu_copilot_preference_data.jsonl
| |-- filtered_preference_data,json
| |-- filtered_sft_data.json
|-- src/
| |-- data_synthesis/
| | |-- sft_generator.py
| | |-- preference_generator.py
| | |-- data_filter.py
| |-- model_training/
| |-- agents/
| |-- utils/
|-- scripts/
| |-- run_sft_generator.sh
| |-- run_preference_generator.sh
|-- models/
|-- docs/
|-- logs/
| |-- sft_generator.log
| |-- preference_generator.log

```

## Acknowledgement
## Citation