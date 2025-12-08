import json
import random
import re
from typing import List, Dict, Any
from datasets import Dataset
import numpy as np
from collections import Counter
import os
from tqdm import tqdm
from openai import OpenAI

class PreferenceGenerator:
    '''
    Synthetic Preference Data using DeepSeek API with incremental saving
    '''
    def __init__(self, sft_dataset: Dataset, api_key=None, model="deepseek-chat"):
        self.sft_dataset = sft_dataset
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1" 
        )
        self.model_name = model

        # Preference templates examples
        self.few_shot_preference_examples = [
            {
                "prompt": "Explain the basic concepts of machine learning",
                "chosen": "Machine learning is a branch of artificial intelligence that enables computers to learn from data and make decisions or predictions without explicit programming. Main types include supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, unsupervised learning discovers patterns in data, and reinforcement learning learns optimal strategies through trial and error.",
                "rejected": "Machine learning is about making computers learn by themselves.",
                "criteria": ["completeness", "depth"],
                "reason": "The preferred answer is more comprehensive and detailed, providing specific classifications and definitions, while the rejected answer is overly simplistic."
            },
            {
                "prompt": "What is overfitting? How to avoid it?",
                "chosen": "Overfitting occurs when a machine learning model performs well on training data but poorly on unseen test data. This typically happens because the model is too complex and learns noise from the training data rather than underlying patterns. Avoidance methods include: 1) Using more training data, 2) Regularization techniques, 3) Cross-validation, 4) Simplifying model complexity, 5) Early stopping.",
                "rejected": "Overfitting means the model learns too well, and can be avoided by reducing data.",
                "criteria": ["accuracy", "helpfulness"],
                "reason": "The preferred answer accurately explains overfitting and provides specific avoidance methods, while the rejected answer contains factual errors (reducing data worsens overfitting)."
            }
        ]

        # Preference criteria examples
        self.preference_criteria = {
            "helpfulness": {
                "description": "How helpful the answer is to the user",
                "indicators": ["directly solves the problem", "provides practical information", "meets user needs"],
                "weight": 0.20
            },
            "accuracy": {
                "description": "Factual accuracy of the answer", 
                "indicators": ["information is correct", "based on reliable sources", "avoids errors"],
                "weight": 0.20
            },
            "completeness": {
                "description": "Comprehensiveness of the answer",
                "indicators": ["covers all aspects of the question", "provides sufficient detail", "avoids omissions"],
                "weight": 0.18
            },
            "clarity": {
                "description": "Clarity and understandability of expression",
                "indicators": ["concise and clear language", "well-organized structure", "clear logic"],
                "weight": 0.15
            },
            "depth": {
                "description": "Depth of analysis", 
                "indicators": ["in-depth analysis of reasons", "demonstrates critical thinking", "provides background knowledge"],
                "weight": 0.15
            },
            "educational_value": {
                "description": "Educational and learning value",
                "indicators": ["teaches concepts effectively", "provides learning insights", "encourages further exploration"],
                "weight": 0.12
            }
        }

        # Degradation strategies for generating low-quality responses
        self.degradation_strategies = [
            "oversimplify",
            "add_errors", 
            "make_vague",
            "add_irrelevant",
            "disorganize",
            "incomplete"
        ]

    def call_llm(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """
        Call DeepSeek API to generate response
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return self._generate_fallback_response(prompt)

    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate fallback response when API fails
        """
        return f"This is a response to: {prompt}. The content would normally be generated by the AI."

    def generate_low_quality_response(self, prompt: str, topic: str, strategy: str) -> str:
        """
        Generate low-quality response using specified degradation strategy
        """
        strategy_prompts = {
            "oversimplify": f"""Provide an overly simplified and superficial response to the following question. 
Remove all technical details, depth, and nuance. Give only the most basic information possible.

Question: {prompt}

Response:""",
            
            "add_errors": f"""Provide a response that contains significant factual errors and misunderstandings about {topic}. 
Include incorrect information, mix up concepts, and demonstrate poor understanding of the topic.

Question: {prompt}

Response:""",
            
            "make_vague": f"""Provide a vague, non-committal response that doesn't actually answer the question properly.
Use hedging language, avoid specifics, and be generally unhelpful.

Question: {prompt}

Response:""",
            
            "add_irrelevant": f"""Provide a response that includes a lot of irrelevant information and tangents.
Start with some relevant content but then go off on unrelated topics.

Question: {prompt}

Response:""",
            
            "disorganize": f"""Provide a poorly organized, confusing response with no clear structure.
Jump between topics randomly and make it difficult to follow your reasoning.

Question: {prompt}

Response:""",
            
            "incomplete": f"""Provide an incomplete response that misses key points and leaves important questions unanswered.
Give only partial information and avoid providing a comprehensive answer.

Question: {prompt}

Response:"""
        }
        
        if strategy in strategy_prompts:
            system_prompt = f"""You are simulating a poorly performing AI assistant. Your task is to generate a low-quality response that demonstrates {strategy}."""
            return self.call_llm(strategy_prompts[strategy], system_prompt, temperature=0.9)
        else:
            # Default to oversimplify strategy
            return self.call_llm(strategy_prompts["oversimplify"], "", temperature=0.9)

    def load_existing_samples(self, jsonl_filepath: str) -> List[Dict[str, Any]]:
        '''
        Load existing samples from JSONL file
        '''
        samples = []
        if os.path.exists(jsonl_filepath):
            try:
                with open(jsonl_filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line.strip()))
                print(f"Loaded {len(samples)} existing samples from {jsonl_filepath}")
            except Exception as e:
                print(f"Error loading existing samples: {e}")
        return samples

    def clear_existing_data(self, jsonl_filepath: str, json_filepath: str):
        '''
        Clear existing data files
        '''
        files_cleared = []
        
        if os.path.exists(jsonl_filepath):
            os.remove(jsonl_filepath)
            files_cleared.append(jsonl_filepath)
            
        if os.path.exists(json_filepath):
            os.remove(json_filepath)
            files_cleared.append(json_filepath)
            
        if files_cleared:
            print(f"Cleared existing data files: {', '.join(files_cleared)}")
        else:
            print("No existing data files found to clear")

    def save_single_sample(self, sample: Dict[str, Any], jsonl_filepath: str, json_filepath: str, all_samples: List[Dict[str, Any]]):
        '''
        Save a single sample to both JSONL and JSON files
        '''
        # Create directory if not exists
        directory = os.path.dirname(jsonl_filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Append to JSONL file
        with open(jsonl_filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Save complete dataset to JSON file
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

    def generate_single_preference_sample(self, base_sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        '''
        Generate a single preference data sample corresponding to a specific SFT sample
        '''
        # Construct prompt (same as SFT sample)
        prompt = base_sample["instruction"]
        if base_sample.get("input", "").strip():
            prompt += "\n" + base_sample["input"]

        topic = base_sample.get("topic", "Unknown")
        difficulty = base_sample.get("difficulty", "medium")

        # Use the SFT output as the chosen answer (high quality)
        chosen_answer = base_sample["output"]

        # Generate low-quality answer using DeepSeek API
        # Use index to ensure deterministic strategy selection for the same SFT sample
        strategy_index = index % len(self.degradation_strategies)
        strategy = self.degradation_strategies[strategy_index]
        
        rejected_answer = self.generate_low_quality_response(prompt, topic, strategy)

        # Generate preference reason
        reason, criteria_used = self._generate_preference_reason(
            prompt, chosen_answer, rejected_answer
        )

        # Calculate quality difference
        quality_diff = self._calculate_quality_difference(chosen_answer, rejected_answer)
        
        # Ensure minimum quality difference
        if quality_diff < 0.3:
            # Try a different strategy if quality difference is too small
            alternative_strategies = [s for s in self.degradation_strategies if s != strategy]
            if alternative_strategies:
                alt_strategy_index = (index + 1) % len(alternative_strategies)
                strategy = alternative_strategies[alt_strategy_index]
                rejected_answer = self.generate_low_quality_response(prompt, topic, strategy)
                quality_diff = self._calculate_quality_difference(chosen_answer, rejected_answer)

        preference_sample = {
            "prompt": prompt,
            "chosen": chosen_answer,
            "rejected": rejected_answer,
            "reason": reason,
            "criteria_used": criteria_used,
            "topic": topic,
            "difficulty": difficulty,
            "quality_difference": quality_diff,
            "degradation_strategy": strategy,
            "sft_index": index  # Add index to track correspondence
        }
        
        return preference_sample

    def _generate_preference_reason(self, prompt: str, chosen: str, rejected: str) -> tuple:
        """
        Generate preference reason
        """
        # Randomly select 2-3 preference criteria
        num_criteria = random.randint(2, 3)
        available_criteria = list(self.preference_criteria.keys())
        criteria_used = random.sample(available_criteria, min(num_criteria, len(available_criteria)))
        
        # Construct reason based on criteria
        criteria_descriptions = {
            "helpfulness": "more helpful and practical",
            "accuracy": "more factually accurate", 
            "completeness": "more comprehensive and complete",
            "clarity": "clearer and better organized",
            "depth": "more analytically deep",
            "educational_value": "of greater educational value"
        }
        
        criteria_text = " and ".join([criteria_descriptions[c] for c in criteria_used])
        
        reason_templates = [
            f"The chosen response is {criteria_text}, providing better educational content.",
            f"Based on {', '.join(criteria_used)} criteria, the preferred answer demonstrates higher quality.",
            f"The rejected answer lacks the {criteria_used[0]} and {criteria_used[1]} found in the preferred response.",
            f"Quality difference is evident in {', '.join(criteria_used)} dimensions, with the chosen answer being superior."
        ]
        
        reason = random.choice(reason_templates)
        
        return reason, criteria_used

    def _calculate_quality_difference(self, chosen: str, rejected: str) -> float:
        """
        Calculate quality difference between chosen and rejected answers
        """
        def calculate_quality_score(response):
            score = 0.0
            
            # 1. Length score (longer, more detailed answers are better)
            length = len(response)
            if length < 50:
                length_score = 0.1
            elif length < 100:
                length_score = 0.3
            elif length < 200:
                length_score = 0.6
            else:
                length_score = 0.9
            score += length_score * 0.25
            
            # 2. Sentence structure score
            sentences = [s for s in response.split('.') if s.strip()]
            if len(sentences) >= 4:
                structure_score = 0.9
            elif len(sentences) >= 2:
                structure_score = 0.6
            else:
                structure_score = 0.2
            score += structure_score * 0.25
            
            # 3. Technical depth score (based on technical terms)
            technical_terms = ['algorithm', 'theory', 'analysis', 'framework', 'methodology', 
                             'principle', 'concept', 'model', 'system', 'structure']
            tech_count = sum(1 for term in technical_terms if term in response.lower())
            tech_score = min(tech_count / 5, 1.0)  # Max 1.0 for 5+ terms
            score += tech_score * 0.3
            
            # 4. Educational quality score
            quality_indicators = ['for example', 'specifically', 'in detail', 'comprehensive',
                                'fundamentally', 'theoretically', 'practically', 'analysis']
            quality_count = sum(1 for indicator in quality_indicators if indicator in response.lower())
            quality_score = min(quality_count / 4, 1.0)
            score += quality_score * 0.2
            
            return min(score, 1.0)
        
        chosen_score = calculate_quality_score(chosen)
        rejected_score = calculate_quality_score(rejected)
        
        difference = chosen_score - rejected_score
        
        return max(difference, 0)  # Ensure non-negative

    def generate_preference_data(self, num_samples: int, jsonl_filepath: str, json_filepath: str, overwrite: bool = False) -> Dataset:
        '''
        Generate preference data with incremental saving and resume capability
        '''
        # Clear existing data if overwrite is True
        if overwrite:
            self.clear_existing_data(jsonl_filepath, json_filepath)
            existing_samples = []
            existing_count = 0
        else:
            # Load existing samples if file exists
            existing_samples = self.load_existing_samples(jsonl_filepath)
            existing_count = len(existing_samples)
        
        # Get all SFT samples in order
        sft_samples = self.sft_dataset.to_list()
        
        # If we have existing samples, find the next SFT index to process
        if existing_samples:
            last_sft_index = max([s.get('sft_index', -1) for s in existing_samples])
            next_index = last_sft_index + 1
        else:
            next_index = 0
        
        # Calculate how many more samples we need
        remaining_samples = max(0, min(num_samples, len(sft_samples))) - next_index
        
        if remaining_samples <= 0:
            print(f"Target number of samples ({num_samples}) already reached or SFT dataset exhausted")
            if len(existing_samples) > num_samples:
                existing_samples = existing_samples[:num_samples]
            return Dataset.from_list(existing_samples)
        
        print(f"Found {existing_count} existing samples, generating {remaining_samples} more samples...")
        print(f"Starting from SFT index {next_index}")
        
        # Progress bar for new samples
        progress_bar = tqdm(total=remaining_samples, desc="Generating preference samples")
        
        # Start from existing samples
        current_samples = existing_samples.copy()
        
        for i in range(remaining_samples):
            if next_index + i >= len(sft_samples):
                print(f"\nWarning: SFT dataset exhausted at index {next_index + i}")
                break
                
            try:
                # Get the corresponding SFT sample in order
                base_sample = sft_samples[next_index + i]
                
                # Generate preference sample
                sample = self.generate_single_preference_sample(base_sample, next_index + i)
                current_samples.append(sample)
                
                # Save immediately to both JSONL and JSON files
                self.save_single_sample(sample, jsonl_filepath, json_filepath, current_samples)
                
                # Update progress
                progress_bar.update(1)
                current_count = len(current_samples)
                progress_bar.set_description(f"Generated {current_count}/{num_samples} samples")
                progress_bar.set_postfix({
                    "topic": sample["topic"],
                    "strategy": sample["degradation_strategy"],
                    "quality_diff": f"{sample['quality_difference']:.2f}",
                    "sft_index": sample["sft_index"]
                })
                
            except Exception as e:
                print(f"Error generating sample for SFT index {next_index + i}: {e}")
                # Skip this sample but continue with next
                continue

        progress_bar.close()
        
        # Ensure we don't exceed the target number
        if len(current_samples) > num_samples:
            current_samples = current_samples[:num_samples]
        
        # Convert to HuggingFace Dataset format
        dataset = Dataset.from_list(current_samples)

        # Print statistics
        self._print_generation_stats(dataset)
        
        return dataset

    def _print_generation_stats(self, dataset: Dataset):
        """Print generation statistics"""
        samples = dataset.to_list()
        
        print(f"\n=== Preference Data Generation Statistics ===")
        print(f"Total samples: {len(samples)}")
        
        # Topic distribution
        topics = [s.get('topic', 'Unknown') for s in samples]
        topic_dist = Counter(topics)
        print(f"\nTopic distribution:")
        for topic, count in topic_dist.most_common():
            print(f"  - {topic}: {count} samples ({count/len(samples)*100:.1f}%)")
        
        # Strategy distribution
        strategies = [s.get('degradation_strategy', 'unknown') for s in samples]
        strategy_dist = Counter(strategies)
        print(f"\nDegradation strategy distribution:")
        for strategy, count in strategy_dist.most_common():
            print(f"  - {strategy}: {count} samples ({count/len(samples)*100:.1f}%)")
        
        # Quality difference statistics
        quality_diffs = [s.get('quality_difference', 0) for s in samples]
        print(f"\nQuality difference statistics:")
        print(f"  - Average difference: {np.mean(quality_diffs):.3f}")
        print(f"  - Minimum difference: {min(quality_diffs):.3f}")
        print(f"  - Maximum difference: {max(quality_diffs):.3f}")
        
        # Count samples with low quality difference
        low_diff_count = sum(1 for d in quality_diffs if d < 0.3)
        medium_diff_count = sum(1 for d in quality_diffs if 0.3 <= d < 0.6)
        high_diff_count = sum(1 for d in quality_diffs if d >= 0.6)
        
        print(f"  - Samples with difference < 0.3: {low_diff_count} ({low_diff_count/len(samples)*100:.1f}%)")
        print(f"  - Samples with difference 0.3-0.6: {medium_diff_count} ({medium_diff_count/len(samples)*100:.1f}%)")
        print(f"  - Samples with difference >= 0.6: {high_diff_count} ({high_diff_count/len(samples)*100:.1f}%)")
        
        # Criteria usage statistics
        all_criteria = []
        for s in samples:
            all_criteria.extend(s.get('criteria_used', []))
        criteria_dist = Counter(all_criteria)
        print(f"\nPreference criteria usage frequency:")
        for criteria, count in criteria_dist.most_common():
            print(f"  - {criteria}: {count} times")
        
        # SFT index range
        sft_indices = [s.get('sft_index', -1) for s in samples]
        print(f"\nSFT index range: {min(sft_indices)} to {max(sft_indices)}")

def load_sft_data(file_path: str) -> Dataset:
    """
    Load SFT data from JSON file
    """
    print(f"Loading SFT data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    
    print(f"Successfully loaded {len(sft_data)} SFT samples")
    
    # Convert to Dataset
    dataset = Dataset.from_list(sft_data)
    return dataset

def get_project_root():
    """
    Get the project root directory based on the current file location
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

def main(overwrite: bool = False):
    """Main function: Generate preference data with incremental saving"""
    
    # Load SFT data from the specified path
    project_root = get_project_root()
    sft_file_path = os.path.join(project_root, "data", "edu_copilot_sft_data.json")
    
    try:
        sft_dataset = load_sft_data(sft_file_path)
        
        # Initialize generator with the loaded SFT data and API credentials
        generator = PreferenceGenerator(
            sft_dataset=sft_dataset,
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            model=os.environ.get("MODEL_NAME", "deepseek-chat")
        )
        
        # Set file paths
        jsonl_filepath = os.path.join(project_root, "data", "edu_copilot_preference_data.jsonl")
        json_filepath = os.path.join(project_root, "data", "edu_copilot_preference_data.json")
        
        # Use the same number of samples as SFT dataset
        num_preference_samples = len(sft_dataset)
        
        print(f"SFT dataset contains {num_preference_samples} samples")
        print(f"Generating {num_preference_samples} preference samples using DeepSeek API...")
        
        mode = "OVERWRITE" if overwrite else "CONTINUE"
        print(f"Mode: {mode}")
        
        # Generate preference data with incremental saving
        preference_dataset = generator.generate_preference_data(
            num_samples=num_preference_samples,
            jsonl_filepath=jsonl_filepath,
            json_filepath=json_filepath,
            overwrite=overwrite
        )
        
        print(f"\n=== Generation Complete ===")
        print(f"Successfully generated {len(preference_dataset)} samples")
        print(f"Incremental data saved to: {jsonl_filepath}")
        print(f"Complete dataset saved to: {json_filepath}")
        
        # Verify correspondence
        if len(preference_dataset) > 0:
            first_sample = preference_dataset[0]
            print(f"\nFirst sample correspondence:")
            print(f"  - SFT index: {first_sample.get('sft_index', 'N/A')}")
            print(f"  - Topic: {first_sample.get('topic', 'N/A')}")
            print(f"  - Prompt: {first_sample.get('prompt', 'N/A')[:100]}...")
        
        return preference_dataset
        
    except FileNotFoundError:
        print(f"Error: SFT data file not found at {sft_file_path}")
        print("Please make sure the file exists and the path is correct.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {sft_file_path}")
        return None
    except Exception as e:
        print(f"Error during preference data generation: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate preference dataset for educational purposes")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing data instead of continuing from existing")
    
    args = parser.parse_args()
    
    print("\n=== Generating Preference Dataset ===")
    print("Note: Make sure DEEPSEEK_API_KEY environment variable is set")
    main(overwrite=args.overwrite)