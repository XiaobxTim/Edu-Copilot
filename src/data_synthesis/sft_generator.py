import json
import random
from typing import List, Dict, Any
from datasets import Dataset
import os
from tqdm import tqdm
from openai import OpenAI

class SFTGenerator:
    '''
    A class to generate Supervised Fine-Tuning (SFT) data for educational purposes.
    '''

    def __init__(self, api_key=None, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1" 
        )
        self.model_name = model

        # seed instructions for reproducibility
        self.seed_instructions = [
            # Basic concept explanation
            {
                "instruction": "Explain the basic concept of {concept}",
                "input": "",
            },
            {
                "instruction": "What is {concept}? Please explain in simple terms",
                "input": "",
            },
            {
                "instruction": "Explain the definition and characteristics of {concept} in detail",
                "input": "",
            },
            
            # Comparative analysis
            {
                "instruction": "Compare the similarities and differences between {concept1} and {concept2}",
                "input": "",
            },
            {
                "instruction": "What are the relationships and differences between {concept1} and {concept2}?",
                "input": "",
            },
            
            # Practical application
            {
                "instruction": "Give an example of {concept} in practical application",
                "input": "",
            },
            {
                "instruction": "How to use {concept} to solve {problem_type} problems?",
                "input": "Specific problem description: {problem_desc}",
            },
            
            # Learning methods
            {
                "instruction": "How to learn {concept} efficiently?",
                "input": "",
            },
            {
                "instruction": "Design a study plan for {concept}",
                "input": "Study duration: {duration}",
            },
            
            # Q&A
            {
                "instruction": "Answer common questions about {concept}: {question}",
                "input": "",
            },
            {
                "instruction": "Solve this {concept}-related problem: {problem}",
                "input": "",
            }
        ]

        # Domain knowledge base
        self.education_knowledge_base = {
            "Mathematics": {
                "concepts": ["Calculus", "Linear Algebra", "Probability Theory", "Statistics", "Discrete Mathematics", "Number Theory", "Topology"],
                "questions": [
                    "How to calculate the derivative of this function?", 
                    "How to find the eigenvalues of this matrix?",
                    "What's the solution to this probability problem?"
                ],
                "problem_types": ["Derivative Problems", "Integration Problems", "Matrix Operations", "Probability Calculations"]
            },
            "Computer Science": {
                "concepts": ["Programming", "Algorithms", "Data Structures", "Machine Learning", "Databases", "Operating Systems", "Computer Networks"],
                "questions": [
                    "What is the time complexity of this algorithm?",
                    "How to optimize the performance of this data structure?",
                    "What is the principle behind this machine learning model?"
                ],
                "problem_types": ["Programming Problems", "Algorithm Design", "System Design", "Debugging Problems"]
            },
            "Physics": {
                "concepts": ["Mechanics", "Electromagnetism", "Thermodynamics", "Quantum Physics", "Relativity", "Optics", "Atomic Physics"],
                "questions": [
                    "What are the application conditions of this physical law?",
                    "How to derive this physical formula?",
                    "How to explain this experimental phenomenon?"
                ],
                "problem_types": ["Mechanics Problems", "Electromagnetism Problems", "Thermodynamics Problems", "Quantum Physics Problems"]
            },
            "Literature": {
                "concepts": ["Novel Analysis", "Poetry Appreciation", "Literary Theory", "Writing Techniques", "Literary History", "Rhetorical Devices"],
                "questions": [
                    "What is the theme of this novel?",
                    "What are the artistic features of this poem?",
                    "What are the characteristics of this literary school?"
                ],
                "problem_types": ["Text Analysis", "Literary Criticism", "Writing Guidance", "Literary History Problems"]
            },
            "History": {
                "concepts": ["Ancient History", "Modern History", "World History", "Cultural History", "Political History", "Economic History"],
                "questions": [
                    "What is the significance of this historical event?",
                    "What are the contributions of this historical figure?",
                    "What are the characteristics of this historical period?"
                ],
                "problem_types": ["Historical Analysis", "Event Evaluation", "Character Assessment", "Period Comparison"]
            }
        }

        # Difficulties levels
        self.difficulty_templates = {
            "easy": {
                "prefix": ["Explain simply", "Basic concept", "Introductory knowledge"],
                "explanation_style": "Use simple and easy-to-understand language, suitable for beginners"
            },
            "medium": {
                "prefix": ["Explain in detail", "In-depth analysis", "Systematic explanation"],
                "explanation_style": "Use more professional terminology and detailed explanations"
            },
            "hard": {
                "prefix": ["Deep dive", "Professional analysis", "Academic research"],
                "explanation_style": "Use professional and in-depth academic language with advanced concepts"
            }
        }

    def call_llm(self, instruction: str, input_text: str, difficulty: str, concept: str, topic: str) -> str:
        """
        Call the LLM API to generate high-quality output
        """
        # Prepare the prompt with specific requirements
        explanation_style = self.difficulty_templates[difficulty]["explanation_style"]
        
        system_prompt = f"""You are an expert educator in {topic}. Provide a high-quality, accurate, and pedagogically sound explanation for the given instruction.

Difficulty Level: {difficulty.upper()}
Style Requirement: {explanation_style}
Concept: {concept}
Topic: {topic}

Please provide a comprehensive, well-structured response that is appropriate for the specified difficulty level."""

        if input_text.strip():
            user_prompt = f"Instruction: {instruction}\nInput: {input_text}\nPlease provide a detailed answer:"
        else:
            user_prompt = f"Instruction: {instruction}\nPlease provide a detailed answer:"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            # Return a fallback response
            return self._generate_fallback_output(instruction, input_text, difficulty, concept, topic)

    def _generate_fallback_output(self, instruction: str, input_text: str, difficulty: str, concept: str, topic: str) -> str:
        """
        Generate a fallback output when API call fails
        """
        base_output = f"This is a {difficulty} level explanation about {concept} in {topic}.\n\n"
        
        if "explain" in instruction.lower() or "what is" in instruction.lower():
            base_output += f"{concept} is an important concept in {topic}. "
            if difficulty == "easy":
                base_output += f"In simple terms, {concept} can be understood as a fundamental idea that helps us understand {topic} better."
            elif difficulty == "medium":
                base_output += f"From a technical perspective, {concept} involves key principles and applications that are essential for understanding {topic}."
            else:
                base_output += f"At an advanced level, {concept} encompasses complex theoretical frameworks and cutting-edge research in {topic}."
        
        elif "compare" in instruction.lower() or "difference" in instruction.lower():
            base_output += f"When comparing concepts in {topic}, it's important to consider both similarities and differences."
        
        elif "example" in instruction.lower() or "application" in instruction.lower():
            base_output += f"Here's a practical example of how {concept} is applied in real-world scenarios."
        
        elif "learn" in instruction.lower() or "study" in instruction.lower():
            base_output += f"Here are effective strategies for learning {concept} efficiently."
        
        return base_output

    def generate_single_sample(self) -> Dict[str, Any]:
        '''
        Generate a single SFT data sample using API for output
        '''
        # Randomly select a topic
        topic = random.choice(list(self.education_knowledge_base.keys()))
        topic_info = self.education_knowledge_base[topic]

        # Randomly select a concept
        concept = random.choice(topic_info["concepts"])
        concept2 = random.choice([c for c in topic_info["concepts"] if c != concept])

        # Randomly select difficulty level
        difficulty = random.choice(list(self.difficulty_templates.keys()))

        # Randomly select a seed template
        seed_template = random.choice(self.seed_instructions)

        # Fill in the template for instruction
        instruction = seed_template["instruction"]
        instruction = instruction.replace("{concept}", concept)
        if "{concept1}" in instruction and "{concept2}" in instruction:
            instruction = instruction.replace("{concept1}", concept).replace("{concept2}", concept2)
        if "{question}" in instruction:
            instruction = instruction.replace("{question}", random.choice(topic_info["questions"]))
        if "{problem}" in instruction:
            instruction = instruction.replace("{problem}", random.choice(topic_info["problems"]))
        instruction = instruction.replace("{topic}", topic)

        # Add difficulty prefix
        difficulty_prefix = random.choice(self.difficulty_templates[difficulty]["prefix"])
        instruction = f"{difficulty_prefix}: {instruction}"

        # Fill in input
        input_text = seed_template["input"]
        if input_text:
            if "{problem_type}" in input_text:
                input_text = input_text.replace("{problem_type}", random.choice(topic_info["problem_types"]))
            
            if "{problem_desc}" in input_text:
                input_text = input_text.replace("{problem_desc}", f"A specific problem example about {concept}")
            
            if "{duration}" in input_text:
                input_text = input_text.replace("{duration}", f"{random.randint(1, 4)} weeks")
            
            if "{question}" in input_text:
                input_text = input_text.replace("{question}", random.choice(topic_info["questions"]))
            
            if "{problem}" in input_text:
                input_text = input_text.replace("{problem}", f"A specific application problem of {concept}")
        else:
            input_text = ""

        # Generate output using API
        output = self.call_llm(instruction, input_text, difficulty, concept, topic)

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "topic": topic,
            "concept": concept,
            "concept2": concept2 if concept2 else "",
            "difficulty": difficulty,
            "template_type": seed_template["instruction"][:20] + "..."
        }

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

    def load_existing_samples(self, filepath: str) -> List[Dict[str, Any]]:
        '''
        Load existing samples from JSONL file
        '''
        samples = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line.strip()))
                print(f"Loaded {len(samples)} existing samples from {filepath}")
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

    def generate_sft_data(self, num_samples: int, jsonl_filepath: str, json_filepath: str, overwrite: bool = False) -> Dataset:
        '''
        Generate specified number of SFT data samples and save incrementally
        
        Args:
            num_samples: Total number of samples to generate
            jsonl_filepath: Path to save the JSONL file
            json_filepath: Path to save the JSON file
            overwrite: Whether to overwrite existing data (True) or continue from existing (False)
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
        
        # Calculate how many more samples we need
        remaining_samples = max(0, num_samples - existing_count)
        
        if remaining_samples == 0:
            print(f"Target number of samples ({num_samples}) already reached in {jsonl_filepath}")
            return Dataset.from_list(existing_samples)
        
        print(f"Found {existing_count} existing samples, generating {remaining_samples} more samples...")
        
        # Progress bar for new samples
        progress_bar = tqdm(total=remaining_samples, desc="Generating samples")
        
        # Start from existing count
        current_samples = existing_samples.copy()
        
        for i in range(remaining_samples):
            try:
                sample = self.generate_single_sample()
                current_samples.append(sample)
                
                # Save immediately to both JSONL and JSON files
                self.save_single_sample(sample, jsonl_filepath, json_filepath, current_samples)
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_description(f"Generated {i+1+existing_count}/{num_samples} samples")
                
            except Exception as e:
                print(f"Error generating sample {i+1+existing_count}: {e}")
                continue

        progress_bar.close()
        
        # Convert to HuggingFace Dataset format
        dataset = Dataset.from_list(current_samples)

        # Print Statistics
        self._print_statistics(dataset)

        return dataset
    
    def _print_statistics(self, dataset: Dataset):
        '''
        Print statistics of the generated dataset
        '''
        samples = dataset.to_list()

        # Topics distribution
        topics = [s['topic'] for s in samples]
        topic_dist = {}
        for topic in topics:
            topic_dist[topic] = topic_dist.get(topic, 0) + 1
        
        # Difficulty distribution
        difficulties = [s['difficulty'] for s in samples]
        diff_dist = {}
        for diff in difficulties:
            diff_dist[diff] = diff_dist.get(diff, 0) + 1
        
        # Input field statistics
        has_input = sum(1 for s in samples if s['input'].strip())
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total samples: {len(samples)}")
        print(f"Samples with input: {has_input} ({has_input/len(samples)*100:.1f}%)")
        print(f"\nTopic distribution:")
        for topic, count in topic_dist.items():
            print(f"  - {topic}: {count} ({count/len(samples)*100:.1f}%)")
        print(f"\nDifficulty distribution:")
        for diff, count in diff_dist.items():
            print(f"  - {diff}: {count} ({count/len(samples)*100:.1f}%)")
        
        # Average lengths
        avg_instruction_len = sum(len(s['instruction']) for s in samples) / len(samples)
        avg_output_len = sum(len(s['output']) for s in samples) / len(samples)
        print(f"\nAverage instruction length: {avg_instruction_len:.1f} characters")
        print(f"Average output length: {avg_output_len:.1f} characters")

def get_project_root():
    """
    Get the project root directory based on the current file location
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

def main(overwrite: bool = False):
    '''
    Generate SFT dataset using API with incremental saving
    
    Args:
        overwrite: Whether to overwrite existing data (True) or continue from existing (False)
    '''
    # Initialize generator with API credentials
    generator = SFTGenerator(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        model=os.environ.get("MODEL_NAME", "deepseek-chat")
    )

    # Set file paths
    project_root = get_project_root()
    jsonl_filepath = os.path.join(project_root, "data", "edu_copilot_sft_data.jsonl")
    json_filepath = os.path.join(project_root, "data", "edu_copilot_sft_data.json")

    # Generate dataset with incremental saving
    num_samples = 5000
    mode = "OVERWRITE" if overwrite else "CONTINUE"
    print(f"Generating {num_samples} samples using API (mode: {mode})...")
    dataset = generator.generate_sft_data(num_samples, jsonl_filepath, json_filepath, overwrite=overwrite)

    print(f"\n=== Generation Complete ===")
    print(f"Successfully generated {len(dataset)} samples")
    print(f"Incremental data saved to: {jsonl_filepath}")
    print(f"Complete dataset saved to: {json_filepath}")
    
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SFT dataset for educational purposes")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing data instead of continuing from existing")
    
    args = parser.parse_args()
    
    print("\n=== Generating Dataset ===")
    print("Note: Make sure DEEPSEEK_API_KEY environment variable is set")
    main(overwrite=args.overwrite)