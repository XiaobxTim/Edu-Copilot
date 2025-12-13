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
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

## Introduction
### Core Feature
### System Architecture​
## Quick Start
**Environment Setup**
```
conda create -n <env_name> python=3.12
conda activate <env_name>
pip install -r requirements.txt -i https://pypi.org/simple
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

**Step 2: Model Training**
```
chmod +x ./scripts/run_sfttrainer.sh
chmod +x ./scripts/run_sfttrainer.sh
chmod +x ./scripts/run_eval.sh

# SFT Trainer
./scripts/run_sfttrainer.sh

# DPO Trainer
./scripts/run_dpotrainer.sh

# Evaluation
export DEEPSEEK_API_KEY="you api key"
./scripts/run_eval.sh
```

## Structure

```plaintext
Edu-Copilot/
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
| |-- eval_data.json
| |-- evaluation_results.json
|-- src/
| |-- data_synthesis/
| | |-- sft_generator.py
| | |-- preference_generator.py
| | |-- data_filter.py
| |-- model_training/
| | |-- SFTTrainer.py
| | |-- DPOTrainer.py
| | |-- Evaluation.py
| |-- agents/
| |-- utils/
|-- scripts/
| |-- run_sft_generator.sh
| |-- run_preference_generator.sh
| |-- run_data_filter.sh
| |-- run_sfttrainer.sh
| |-- run_dpotrainer.sh
| |-- run_eval.sh
|-- models/
| |-- sft_model/best_model
| |-- dpo_model/best_model
|-- docs/
|-- logs/
| |-- sft_generator.log
| |-- preference_generator.log
| |-- sfttrainer.log
| |-- dpotrainer.log
| |-- data_filter.log
| |-- evaluation.log

```

## Acknowledgements

This project represents a practical application of my learning in natural language processing and machine learning. I would like to acknowledge the following contributions:

**Technical Support**:
- Hugging Face for the Transformers library and pre-trained models
- The Sentence Transformers development team for making text similarity computation accessible
- The broader open-source AI/ML community

**Learning Resources**:
- Online educational platforms (Coursera, deeplearning.ai, etc.)
- The research community for sharing knowledge through papers and tutorials
- Technical documentation and blog posts that provided valuable insights

**Code References**:
While all code in this repository was written by me, I consulted and learned from various open-source projects, including (in alphabetical order):
- Hugging Face Transformers
- Sentence Transformers
- Numerous educational GitHub repositories

**Community Support**:
Stack Overflow, GitHub Discussions, and various technical forums where developers generously share their knowledge.

**Personal**:
To myself, for maintaining curiosity and persistence in learning and building.

---
*This project was independently designed, developed, and tested by me.*

## Citation
```
@inproceedings{tan-etal-2024-large,
    title = "Large Language Models for Data Annotation and Synthesis: A Survey",
    author = "Tan, Zhen  and
      Li, Dawei  and
      Wang, Song  and
      Beigi, Alimohammad  and
      Jiang, Bohan  and
      Bhattacharjee, Amrita  and
      Karami, Mansooreh  and
      Li, Jundong  and
      Cheng, Lu  and
      Liu, Huan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.54/",
    doi = "10.18653/v1/2024.emnlp-main.54",
    pages = "930--957",
    abstract = "Data annotation and synthesis generally refers to the labeling or generating of raw data with relevant information, which could be used for improving the efficacy of machine learning models. The process, however, is labor-intensive and costly. The emergence of advanced Large Language Models (LLMs), exemplified by GPT-4, presents an unprecedented opportunity to automate the complicated process of data annotation and synthesis. While existing surveys have extensively covered LLM architecture, training, and general applications, we uniquely focus on their specific utility for data annotation. This survey contributes to three core aspects: LLM-Based Annotation Generation, LLM-Generated Annotations Assessment, and LLM-Generated Annotations Utilization. Furthermore, this survey includes an in-depth taxonomy of data types that LLMs can annotate, a comprehensive review of learning strategies for models utilizing LLM-generated annotations, and a detailed discussion of the primary challenges and limitations associated with using LLMs for data annotation and synthesis. Serving as a key guide, this survey aims to assist researchers and practitioners in exploring the potential of the latest LLMs for data annotation, thereby fostering future advancements in this critical field."
}
@misc{singh2025fspofewshotpreferenceoptimization,
      title={FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users}, 
      author={Anikait Singh and Sheryl Hsu and Kyle Hsu and Eric Mitchell and Stefano Ermon and Tatsunori Hashimoto and Archit Sharma and Chelsea Finn},
      year={2025},
      eprint={2502.19312},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.19312}, 
}
@misc{yasunaga2024almaalignmentminimalannotation,
      title={ALMA: Alignment with Minimal Annotation}, 
      author={Michihiro Yasunaga and Leonid Shamis and Chunting Zhou and Andrew Cohen and Jason Weston and Luke Zettlemoyer and Marjan Ghazvininejad},
      year={2024},
      eprint={2412.04305},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.04305}, 
}
@misc{he2025airsystematicanalysisannotations,
      title={AIR: A Systematic Analysis of Annotations, Instructions, and Response Pairs in Preference Dataset}, 
      author={Bingxiang He and Wenbin Zhang and Jiaxi Song and Cheng Qian and Zixuan Fu and Bowen Sun and Ning Ding and Haiwen Hong and Longtao Huang and Hui Xue and Ganqu Cui and Wanxiang Che and Zhiyuan Liu and Maosong Sun},
      year={2025},
      eprint={2504.03612},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.03612}, 
}
@inproceedings{morimura-etal-2024-filtered,
    title = "Filtered Direct Preference Optimization",
    author = "Morimura, Tetsuro  and
      Sakamoto, Mitsuki  and
      Jinnai, Yuu  and
      Abe, Kenshi  and
      Ariu, Kaito",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1266/",
    doi = "10.18653/v1/2024.emnlp-main.1266",
    pages = "22729--22770",
    abstract = "Reinforcement learning from human feedback (RLHF) plays a crucial role in aligning language models with human preferences. While the significance of dataset quality is generally recognized, explicit investigations into its impact within the RLHF framework, to our knowledge, have been limited. This paper addresses the issue of text quality within the preference dataset by focusing on direct preference optimization (DPO), an increasingly adopted reward-model-free RLHF method. We confirm that text quality significantly influences the performance of models optimized with DPO more than those optimized with reward-model-based RLHF. Building on this new insight, we propose an extension of DPO, termed filtered direct preference optimization (fDPO). fDPO uses a trained reward model to monitor the quality of texts within the preference dataset during DPO training. Samples of lower quality are discarded based on comparisons with texts generated by the model being optimized, resulting in a more accurate dataset. Experimental results demonstrate that fDPO enhances the final model performance. Our code is available at https://github.com/CyberAgentAILab/filtered-dpo."
}
@misc{albalak2024surveydataselectionlanguage,
      title={A Survey on Data Selection for Language Models}, 
      author={Alon Albalak and Yanai Elazar and Sang Michael Xie and Shayne Longpre and Nathan Lambert and Xinyi Wang and Niklas Muennighoff and Bairu Hou and Liangming Pan and Haewon Jeong and Colin Raffel and Shiyu Chang and Tatsunori Hashimoto and William Yang Wang},
      year={2024},
      eprint={2402.16827},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.16827}, 
}
@misc{liu2024makesgooddataalignment,
      title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning}, 
      author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
      year={2024},
      eprint={2312.15685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.15685}, 
}
@misc{chu2025llmagentseducationadvances,
      title={LLM Agents for Education: Advances and Applications}, 
      author={Zhendong Chu and Shen Wang and Jian Xie and Tinghui Zhu and Yibo Yan and Jinheng Ye and Aoxiao Zhong and Xuming Hu and Jing Liang and Philip S. Yu and Qingsong Wen},
      year={2025},
      eprint={2503.11733},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2503.11733}, 
}
@misc{zhang2025eduplannerllmbasedmultiagentsystems,
      title={EduPlanner: LLM-Based Multi-Agent Systems for Customized and Intelligent Instructional Design}, 
      author={Xueqiao Zhang and Chao Zhang and Jianwen Sun and Jun Xiao and Yi Yang and Yawei Luo},
      year={2025},
      eprint={2504.05370},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.05370}, 
}
@misc{zhang2024simulatingclassroomeducationllmempowered,
      title={Simulating Classroom Education with LLM-Empowered Agents}, 
      author={Zheyuan Zhang and Daniel Zhang-Li and Jifan Yu and Linlu Gong and Jinchang Zhou and Zhanxin Hao and Jianxiao Jiang and Jie Cao and Huiqin Liu and Zhiyuan Liu and Lei Hou and Juanzi Li},
      year={2024},
      eprint={2406.19226},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.19226}, 
}
```