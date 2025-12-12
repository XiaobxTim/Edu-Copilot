import torch
import numpy as np
import json
import logging
import os
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EducationalDPOTrainer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        self.sft_model_path = Path(__file__).parent.parent / 'models' / 'sft_model' / 'best_model'
        self.model_name = str(self.sft_model_path) if self.sft_model_path.exists() else model_name
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_model_and_tokenizer(self):
        """加载 SFT 后的模型作为 DPO 的基础模型和参考模型"""
        logger.info(f"Loading model and tokenizer from: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # 推荐 DPO 使用左填充
        
        # 加载 SFT 后的模型作为 policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        # 加载参考模型 (通常是 SFT 模型的一个副本，但在 DPO 期间保持冻结)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.ref_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.ref_model.config.eos_token_id = self.tokenizer.eos_token_id

        logger.info("Models and tokenizer loaded successfully")
        
    def load_and_split_data(self, data_path, train_ratio=0.9, max_samples=None):
        """加载和拆分偏好数据"""
        logger.info(f"Loading DPO preference data: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples and len(data) > max_samples:
            logger.info(f"Limiting to {max_samples} samples")
            data = data[:max_samples]
        
        # DPO 数据格式要求每个样本包含 'prompt', 'chosen', 'rejected'
        # 假设您的偏好数据格式为：
        # [{"prompt": "...", "chosen": "...", "rejected": "...", "id": "..."}]
        formatted_data = []
        for item in data:
            # 这里的键名需要与您在 Q1(2) 中生成的数据格式一致
            # 示例中使用了 Alpaca/Instruction-Tuning 风格的格式化
            instruction = item.get("prompt", "")
            input_text = item.get("input", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            
            if not instruction or not chosen or not rejected:
                logger.warning("Missing required fields in preference data, skipping.")
                continue

            if input_text:
                prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

            formatted_data.append({
                'prompt': prompt, 
                'chosen': chosen, 
                'rejected': rejected
            })

        if len(formatted_data) == 0:
            raise ValueError("No valid DPO preference data samples found")

        dataset = Dataset.from_list(formatted_data)
        split_dataset = dataset.train_test_split(
            train_size=train_ratio, 
            shuffle=True, 
            seed=42
        )
        
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        logger.info(f"DPO Data loaded successfully")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Eval samples: {len(self.eval_dataset)}")
        
    def setup_training_args(self):
        """设置 DPO 训练参数"""
        DIR = Path(__file__).parent.resolve()
        SAVE_PATH = Path("/root/autodl-tmp/dpo_model/best_model")
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

        # 使用 DPOConfig 或 TrainingArguments
        training_args = DPOConfig(
            output_dir=str(SAVE_PATH),
            overwrite_output_dir=True,
            
            # 训练参数
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            
            # 学习率
            learning_rate=5e-7, # DPO 的学习率通常比 SFT 低
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            
            # 优化
            optim="adamw_torch",
            max_grad_norm=1.0,
            
            # 评估和保存策略
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="loss", # DPO 的指标通常是 loss
            greater_is_better=False,
            
            # 其他
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
            remove_unused_columns=False, # 必须为 False, 否则 DPO Trainer 可能会删除 prompt, chosen, rejected
        )
        
        return training_args
    
    def train(self, data_path, max_samples=None):
        """运行 DPO 训练"""
        # 1. 加载模型和分词器
        self.load_model_and_tokenizer()
        
        # 2. 加载和拆分数据
        self.load_and_split_data(data_path, max_samples=max_samples)
        
        # 3. 设置训练参数
        training_args = self.setup_training_args()
        
        # 定义长度参数，并计算 max_target_length
        MAX_LENGTH = 128
        MAX_PROMPT_LENGTH = 64
        MAX_TARGET_LENGTH = MAX_LENGTH - MAX_PROMPT_LENGTH # 计算 Target 的最大长度

        # 4. 初始化 DPO Trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model, # SFT 后的模型副本
            args=training_args,
            beta=0.1,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_target_length=MAX_TARGET_LENGTH, 
            label_pad_token_id=self.tokenizer.pad_token_id, # 最终修复
        )
        
        # 5. 开始训练
        logger.info("Begin DPO training with evaluation ...")
        trainer.train()
        
        # 6. 保存最佳模型
        DIR = Path(__file__).parent.resolve()
        SAVE_PATH = Path("/root/autodl-tmp/dpo_model/best_model")
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("DPO training done, saving best model ...")
        trainer.save_model(str(SAVE_PATH))
        self.tokenizer.save_pretrained(str(SAVE_PATH))
        
        # 7. 评估最终模型
        logger.info("Evaluating final DPO model ...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

def main():
    """主函数，运行 DPO 训练"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    trainer = EducationalDPOTrainer()
    
    DIR = Path(__file__).parent.resolve()
    # 假设您的偏好数据保存在 filtered_preference_data.json
    DPO_PATH = DIR.parent.parent / 'data' / 'filtered_preference_data.json'
    
    if not os.path.exists(DPO_PATH):
        logger.error(f"DPO Data file not found: {DPO_PATH}")
        return
    
    try:
        # 使用 Q1(4) 中过滤后的 3k 样本进行训练
        trainer.train(DPO_PATH, max_samples=3000)
    except Exception as e:
        logger.error(f"An error occurred during DPO training: {e}")
        return

if __name__ == "__main__":
    # 确保在运行 DPO 之前，SFT 训练已经完成并保存了模型
    main()