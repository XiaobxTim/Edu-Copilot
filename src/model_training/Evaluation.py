import torch
import json
import logging
import os
import requests
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths and Config ---
# Ensure these paths correctly point to your saved SFT and DPO model directories.
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
# Assuming SFT model is saved here (used as the reliable source for the tokenizer)
SFT_MODEL_PATH = ROOT_DIR / 'models' / 'sft_model' / 'best_model' 
# Assuming DPO model is saved here
DPO_MODEL_PATH = ROOT_DIR / 'models' / 'dpo_model' / 'best_model' 

# Evaluation data path
EVAL_DATA_PATH = ROOT_DIR / 'data' / 'eval_data.json'

# Deepseek API parameters
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_JUDGE_MODEL = "deepseek-v3"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def load_models_and_tokenizer():
    """Load SFT and DPO models along with the tokenizer."""
    
    # FIX: Load tokenizer from SFT path as DPO path might be corrupted.
    SFT_PATH_FOR_TOKENIZER = str(SFT_MODEL_PATH)
    logger.info(f"Loading tokenizer from SFT Model Path: {SFT_PATH_FOR_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_PATH_FOR_TOKENIZER, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FIX: Force float32 to solve nan/inf numerical instability during generation.
    dtype = torch.float32

    logger.info(f"Loading SFT model from: {SFT_MODEL_PATH} with dtype: {dtype}")
    sft_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_MODEL_PATH),
        torch_dtype=dtype, 
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).eval()

    logger.info(f"Loading DPO model from: {DPO_MODEL_PATH} with dtype: {dtype}")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        str(DPO_MODEL_PATH),
        torch_dtype=dtype, 
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).eval()
    
    return tokenizer, sft_model, dpo_model

def generate_response(model, tokenizer, prompt: str):
    """
    Run model inference to generate a response for the given prompt.
    """
    # Make sure the prompt is formatted correctly for Qwen's chat template
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize, ensuring padding and truncation for batching (even though we process one by one)
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    
    input_ids = inputs["input_ids"].to(model.device)
    # FIX: Explicitly pass attention_mask to prevent generation errors when pad_token == eos_token
    attention_mask = inputs["attention_mask"].to(model.device) 

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask, 
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and strip the prompt part
    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    return response

def get_judge_prompt(user_prompt: str, sft_response: str, dpo_response: str) -> str:
    """Build the prompt for the Deepseek Judge model."""
    return f"""
You are an expert educational LLM evaluation specialist. Your task is to assess the quality of responses provided by two educational assistant models (SFT Model vs DPO Model) for a given user query.

[Scoring Rubric]
1. Accuracy: 0-5 points. Factual correctness of the answer.
2. Difficulty Match: 0-5 points. Does the answer match the difficulty level requested in the Prompt (e.g., college freshman, medium difficulty)?
3. Educational Value: 0-5 points. Is the answer insightful and does it help the student deepen their understanding?

[Task and Responses]
User Prompt: {user_prompt}

SFT Model Response: {sft_response}

DPO Model Response: {dpo_response}

[Evaluation Requirements]
Please score the SFT Model and the DPO Model independently based on the above criteria, and provide a short summary.
The response MUST be a strict JSON object following this structure:
{{
    "SFT_Model": {{
        "Accuracy": <integer from 0-5>,
        "Difficulty_Match": <integer from 0-5>,
        "Educational_Value": <integer from 0-5>
    }},
    "DPO_Model": {{
        "Accuracy": <integer from 0-5>,
        "Difficulty_Match": <integer from 0-5>,
        "Educational_Value": <integer from 0-5>
    }},
    "Summary": "Briefly summarize which model is better and why."
}}
Do not include any explanation or extra text outside of the required JSON object.
"""

def call_deepseek_judge(prompt: str):
    """Request Deepseek API to evaluate the responses."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY unset in environment variables.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": DEEPSEEK_JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0, # Low temperature for deterministic evaluation
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        judge_content = response.json()['choices'][0]['message']['content']
        try:
            # Clean up potential markdown wrappers (e.g., ```json ... ```)
            if judge_content.startswith("```json"):
                judge_content = judge_content.strip().strip("`json").strip("`").strip()
            return json.loads(judge_content)
        except json.JSONDecodeError:
            logger.error(f"Deepseek Judge returned invalid JSON format: {judge_content}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Deepseek API request error: {e}")
        return None

def main_evaluation():
    # Pre-flight checks
    if not SFT_MODEL_PATH.exists() or not DPO_MODEL_PATH.exists():
        logger.error("Cannot find SFT or DPO model directories. Please check model paths.")
        return

    if not os.path.exists(EVAL_DATA_PATH):
        logger.error(f"Cannot find evaluation file: {EVAL_DATA_PATH}")
        return

    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    if len(eval_data) < 5:
        logger.warning(f"Less than 5 evaluation samples found ({len(eval_data)}). Consider adding more samples for robust evaluation.")

    try:
        tokenizer, sft_model, dpo_model = load_models_and_tokenizer()
    except Exception as e:
        # Catch the specific error related to tokenizer loading
        logger.error(f"Model/Tokenizer loading failed: {e}")
        return
        
    all_results = []

    logger.info("Starting LLM-as-a-Judge evaluation...")
    
    for item in eval_data:
        # Construct the full prompt
        prompt = item['prompt'] + (f"\n{item['input']}" if item.get('input') else "")
        
        logger.info(f"--- Sample ID: {item['id']} ---")
        logger.info(f"Prompt: {prompt[:50]}...")

        try:
            # 1. Model generation
            sft_response = generate_response(sft_model, tokenizer, prompt)
            dpo_response = generate_response(dpo_model, tokenizer, prompt)
        except RuntimeError as e:
            logger.error(f"Model generation failed (RuntimeError): {e}")
            sft_response = "Generation Failed"
            dpo_response = "Generation Failed"
            judge_result = {"error": "Generation Failed due to RuntimeError"}
        else:
            logger.info("Responses Generated. Calling Judge API...")
            
            # 2. Call Judge API
            judge_prompt = get_judge_prompt(prompt, sft_response, dpo_response)
            judge_result = call_deepseek_judge(judge_prompt)

        result = {
            "id": item['id'],
            "prompt": prompt,
            "sft_response": sft_response,
            "dpo_response": dpo_response,
            "judge_result": judge_result
        }
        all_results.append(result)

        if judge_result and 'error' not in judge_result:
            sft_total = sum(judge_result['SFT_Model'].values()) if 'SFT_Model' in judge_result and isinstance(judge_result['SFT_Model'], dict) else 'N/A'
            dpo_total = sum(judge_result['DPO_Model'].values()) if 'DPO_Model' in judge_result and isinstance(judge_result['DPO_Model'], dict) else 'N/A'
            logger.info(f"Judge results: SFT Total={sft_total}, DPO Total={dpo_total}")

    # 4. Save results
    output_path = ROOT_DIR / "data" / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Evaluation finished! Results saved to: {output_path}")
    
if __name__ == "__main__":
    main_evaluation()