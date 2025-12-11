import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
)
from trl import SFTTrainer
from datasets import Dataset
import json
import logging
from pathlib import Path
import os

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EducationalSFTTrainer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        
    def load_model_and_tokenizer(self):
        """Loading model and tokenizer"""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
        
    def load_sft_data(self, data_path):
        """Loading SFT data"""
        logger.info(f"Loading SFT data: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            formatted_text = self._format_training_text(item)
            formatted_data.append({'text': formatted_text})
        
        self.train_dataset = Dataset.from_list(formatted_data)
        logger.info(f"SFT loaded successfully, total {len(self.train_dataset)} samples")
        
    def _format_training_text(self, item: dict) -> str:
        """Formatting training text for educational domain"""
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]
        
        # Merge instruction and input if input exists
        if input_text:
            user_message = f"{instruction}\n{input_text}"
        else:
            user_message = instruction
            
        # Format as dialogue
        formatted_text = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        
        return formatted_text
    
    def setup_training_args(self):
        """Setting up training arguments"""
        DIR = Path(__file__).parent.resolve()
        SAVE_PATH = DIR.parent.parent / "models"
        training_args = TrainingArguments(
            output_dir=SAVE_PATH,
            overwrite_output_dir=True,
            num_train_epochs=100,
            learning_rate=2e-5,
            warmup_steps=5,
            logging_steps=10,
            save_steps=10,
            eval_strategy="no",
            save_strategy="steps",
        )
        
        return training_args
    
    def train(self, data_path):
        """Run SFT training"""
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load SFT data
        self.load_sft_data(data_path)
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Initialize SFT Trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
        )
        
        # Begin training
        logger.info("Begin training ...")
        trainer.train()
        
        # Save the trained model and tokenizer
        logger.info("Training done, saved model ...")
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info(f"Model save to: {training_args.output_dir}")
        

def main():
    """Main function to run the SFT training`"""
    # Initialize trainer
    trainer = EducationalSFTTrainer()

    DIR = Path(__file__).parent.resolve()
    SFT_PATH = DIR.parent.parent / 'data' / 'filtered_sft_data.json'
    
    try:
        # Start training
        trainer.train(SFT_PATH)
        
    except Exception as e:
        logger.error(f"Having errors: {e}")
        raise

if __name__ == "__main__":
    main()