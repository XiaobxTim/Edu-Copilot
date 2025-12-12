import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer
from datasets import Dataset, load_dataset
import json
import logging
from pathlib import Path
import os
from sklearn.metrics import accuracy_score

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EducationalSFTTrainer:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def load_model_and_tokenizer(self):
        """Loading model and tokenizer"""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.model.gradient_checkpointing_enable()
        
        logger.info("Model and tokenizer loaded successfully")
        
    def load_and_split_data(self, data_path, train_ratio=0.9, max_samples=None):
        """Loading and splitting SFT data into train and eval sets"""
        logger.info(f"Loading SFT data: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples and len(data) > max_samples:
            logger.info(f"Limiting to {max_samples} samples")
            data = data[:max_samples]
        
        formatted_data = []
        for item in data:
            try:
                formatted_text = self._format_training_text(item)
                formatted_data.append({'text': formatted_text})
            except KeyError as e:
                logger.warning(f"Missing key in data item: {e}, skipping this sample")
                continue
        
        logger.info(f"Successfully formatted {len(formatted_data)} samples")
        
        if len(formatted_data) == 0:
            raise ValueError("No valid data samples found after formatting")
        
        # Split into train and eval
        dataset = Dataset.from_list(formatted_data)
        split_dataset = dataset.train_test_split(
            train_size=train_ratio, 
            shuffle=True, 
            seed=42
        )
        
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        logger.info(f"Data loaded successfully")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Eval samples: {len(self.eval_dataset)}")
        
    def _format_training_text(self, item: dict) -> str:
        """Formatting training text for educational domain"""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if not instruction or not output:
            logger.warning(f"Missing instruction or output in item: {item.get('id', 'unknown')}")
            
        if input_text:
            user_message = f"{instruction}\n{input_text}"
        else:
            user_message = instruction
            
        formatted_text = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        
        return formatted_text
    
    def compute_metrics(self, eval_pred):
        """Compute accuracy metric - fixed to handle list inputs"""
        predictions, labels = eval_pred
        
        if predictions is None or labels is None:
            return {"accuracy": 0.0}
        
        # Convert predictions to numpy array if it's a list
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        if isinstance(labels, list):
            labels = np.array(labels)
        
        if len(predictions.shape) == 3:
            predictions = np.argmax(predictions, axis=-1)
        
        if predictions.shape != labels.shape:
            min_shape = min(predictions.shape[-1], labels.shape[-1]) if len(predictions.shape) == len(labels.shape) else -1
            if min_shape > 0:
                predictions = predictions[..., :min_shape]
                labels = labels[..., :min_shape]
            else:
                return {"accuracy": 0.0, "error": "shape_mismatch"}
        
        mask = labels != -100
        
        predictions_flat = predictions[mask].flatten()
        labels_flat = labels[mask].flatten()
        
        if len(predictions_flat) > 0 and len(labels_flat) > 0:
            min_len = min(len(predictions_flat), len(labels_flat))
            predictions_flat = predictions_flat[:min_len]
            labels_flat = labels_flat[:min_len]
            
            correct = np.sum(predictions_flat == labels_flat)
            total = len(predictions_flat)
            accuracy = float(correct) / float(total) if total > 0 else 0.0
        else:
            accuracy = 0.0
        
        return {"accuracy": accuracy}
    
    def tokenize_function(self, examples):
        """Tokenize text data"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=128,
            return_tensors=None, 
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_datasets(self):
        """Prepare tokenized datasets"""
        if self.train_dataset is None or self.eval_dataset is None:
            raise ValueError("Datasets not loaded")
        
        tokenized_train = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        tokenized_eval = self.eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names,
            desc="Tokenizing evaluation data"
        )
        
        return tokenized_train, tokenized_eval
    
    def setup_training_args(self):
        """Setting up training arguments with evaluation"""
        DIR = Path(__file__).parent.resolve()
        SAVE_PATH = DIR.parent.parent / 'models' / 'sft_model'
        training_args = TrainingArguments(
            output_dir=str(SAVE_PATH),
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            
            # Learning rate
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=50,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            
            # Optimization
            optim="adamw_torch",
            max_grad_norm=1.0,
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            
            eval_accumulation_steps=1,
            eval_do_concat_batches=False,
            
            # Saving
            save_total_limit=3,
            save_only_model=False,
            
            # Logging
            logging_strategy="steps",
            logging_steps=10,
            logging_first_step=True,
            report_to="none",
            
        )
        
        return training_args
    
    def train(self, data_path, max_samples=None):
        """Run SFT training with evaluation"""
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load and split data
        self.load_and_split_data(data_path, max_samples=max_samples)
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize SFT Trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=128,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Begin training
        logger.info("Begin training with evaluation ...")
        
        # Train with evaluation
        trainer.train()
        
        # Save the best model
        DIR = Path(__file__).parent.resolve()
        SAVE_PATH = DIR.parent.parent / 'models' / 'sft_model' / 'best_model'
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("Training done, saving best model ...")
        trainer.save_model(str(SAVE_PATH))
        self.tokenizer.save_pretrained(str(SAVE_PATH))
        
        # Evaluate the final model
        logger.info("Evaluating final model ...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

def main():
    """Main function to run the SFT training"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize trainer
    trainer = EducationalSFTTrainer()
    
    DIR = Path(__file__).parent.resolve()
    SFT_PATH = DIR.parent.parent / 'data' / 'filtered_sft_data.json'
    
    # Start training with a small sample first
    logger.info("Starting training with evaluation...")
    
    if not os.path.exists(SFT_PATH):
        logger.error(f"Data file not found: {SFT_PATH}")
        return
    
    try:
        with open(SFT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Data file loaded successfully, {len(data)} samples found")
        
        for i in range(min(3, len(data))):
            logger.info(f"Sample {i} keys: {list(data[i].keys())}")
            
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        return
    
    trainer.train(SFT_PATH, max_samples=3000)

if __name__ == "__main__":
    main()