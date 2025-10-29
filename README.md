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

# sft data generator
./scripts/run_sft_generator.sh
```

## Structure

```plaintext
edu-copilot/
|-- README.md
|-- check_env.py
|-- requirements.txt
|-- data/
| |--edu_copilot_sft_data.json
| |--edu_copilot_sft_data.jsonl
|-- src/
| |-- data_synthesis/
| | |-- sft_generator.py
| | |-- preference_generator.py
| | |-- data_filter.py
|-- scripts/
| |-- run_sft_generator.sh

```

## Acknowledgement
## Citation