import json
import random
from typing import List, Dict, Any
from datasets import Dataset
import os

class SFTGenerator:
    '''
    A class to generate Supervised Fine-Tuning (SFT) data for educational purposes.
    '''

    def __init__(self):
        # seed instructions for reproducibility
        self.seed_instructions = [
            # Basic concept explanation
            {
                "instruction": "Explain the basic concept of {concept}",
                "input": "",
                "output": "{concept} is an important concept in the field of {topic}, which refers to..."
            },
            {
                "instruction": "What is {concept}? Please explain in simple terms",
                "input": "",
                "output": "Simply put, {concept} is..."
            },
            {
                "instruction": "Explain the definition and characteristics of {concept} in detail",
                "input": "",
                "output": "The definition of {concept} is... Its main characteristics include..."
            },
            
            # Comparative analysis
            {
                "instruction": "Compare the similarities and differences between {concept1} and {concept2}",
                "input": "",
                "output": "{concept1} and {concept2} are both important concepts in {topic}.\nSimilarities:...\nDifferences:..."
            },
            {
                "instruction": "What are the relationships and differences between {concept1} and {concept2}?",
                "input": "",
                "output": "The relationship between {concept1} and {concept2} lies in... The main differences are..."
            },
            
            # Practical application
            {
                "instruction": "Give an example of {concept} in practical application",
                "input": "",
                "output": "A typical application of {concept} in practice is..."
            },
            {
                "instruction": "How to use {concept} to solve {problem_type} problems?",
                "input": "Specific problem description: {problem_desc}",
                "output": "Steps to solve {problem_type} problems using {concept}:\n1. ...\n2. ...\n3. ..."
            },
            
            # Learning methods
            {
                "instruction": "How to learn {concept} efficiently?",
                "input": "",
                "output": "Efficient methods for learning {concept} include:\n1. ...\n2. ...\n3. ..."
            },
            {
                "instruction": "Design a study plan for {concept}",
                "input": "Study duration: {duration}",
                "output": "A {duration} study plan for {concept}:\nWeek 1:...\nWeek 2:...\nWeek 3:..."
            },
            
            # Q&A
            {
                "instruction": "Answer common questions about {concept}: {question}",
                "input": "",
                "output": "Regarding this question about {concept}, the answer is..."
            },
            {
                "instruction": "Solve this {concept}-related problem: {problem}",
                "input": "",
                "output": "Let's solve this {concept} problem. First... then... finally..."
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
                "problem_types": ["Derivative Problems", "Integration Problems", "Matrix Operations", "Probability Calculations"]  # 修复：problems_type -> problem_types
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
                "explanation_style": "Simple and easy-to-understand language"
            },
            "medium": {
                "prefix": ["Explain in detail", "In-depth analysis", "Systematic explanation"],
                "explanation_style": "More professional terminology"
            },
            "hard": {
                "prefix": ["Deep dive", "Professional analysis", "Academic research"],
                "explanation_style": "Professional and in-depth academic language"
            }
        }

    def generate_single_sample(self) -> Dict[str, Any]:
        '''
        Generate a single SFT data sample
        '''
        # Randomly select a topic
        topic = random.choice(list(self.education_knowledge_base.keys()))
        topic_info = self.education_knowledge_base[topic]

        # Randomly select a concept
        concept = random.choice(topic_info["concepts"])

        # 50% chance to select a second concept for comparative analysis
        concept2 = None
        if random.random() > 0.5 and len(topic_info["concepts"]) > 1:
            concept2 = random.choice([c for c in topic_info["concepts"] if c != concept])

        # Randomly select difficulty level
        difficulty = random.choice(list(self.difficulty_templates.keys()))

        # Randomly select a seed template
        seed_template = random.choice(self.seed_instructions)

        # Fill in the template
        instruction = seed_template["instruction"]
        if concept2:
            instruction = instruction.replace("{concept1}", concept).replace("{concept2}", concept2).replace("{topic}", topic)
        else:
            instruction = instruction.replace("{concept}", concept).replace("{topic}", topic)

        # Add difficulty prefix
        difficulty_prefix = random.choice(self.difficulty_templates[difficulty]["prefix"])
        instruction = f"{difficulty_prefix}: {instruction}"

        # Fill in input
        input_text = seed_template["input"]
        if input_text:
            # Fill input template
            if "{problem_type}" in input_text:
                problem_type = random.choice(topic_info["problem_types"])
                input_text = input_text.replace("{problem_type}", problem_type)
            
            if "{problem_desc}" in input_text:
                problem_desc = f"A specific problem example about {concept}"
                input_text = input_text.replace("{problem_desc}", problem_desc)
            
            if "{duration}" in input_text:
                duration = f"{random.randint(1, 4)} weeks"
                input_text = input_text.replace("{duration}", duration)
            
            if "{question}" in input_text:
                question = random.choice(topic_info["questions"])
                input_text = input_text.replace("{question}", question)
            
            if "{problem}" in input_text:
                problem = f"A specific application problem of {concept}"
                input_text = input_text.replace("{problem}", problem)
        else:
            input_text = ""

        # Generate output
        output = self._generate_output(seed_template["output"], concept, concept2, topic, difficulty)

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "topic": topic,
            "concept": concept,
            "concept2": concept2 if concept2 else "",
            "difficulty": difficulty,
            "template_type": seed_template["instruction"][:20] + "..."  # Record template type used
        }
    
    def _generate_output(self, output_template: str, concept: str, concept2: str, topic: str, difficulty: str) -> str:
        '''
        Generate output text based on the template and parameters
        '''
        output = output_template

        # Replace placeholders
        output = output.replace("{concept}", concept).replace("{topic}", topic)
        if concept2:
            output = output.replace("{concept1}", concept).replace("{concept2}", concept2)
        
        # Adjust output style and content depth based on difficulty
        output = self._adjust_output_by_difficulty(output, concept, difficulty)

        # Replace other placeholders if any
        if "{problem_type}" in output:
            problem_type = random.choice(self.education_knowledge_base[topic]["problem_types"])
            output = output.replace("{problem_type}", problem_type)

        if "{problem_desc}" in output:
            problem_desc = f"A specific problem example about {concept}"
            output = output.replace("{problem_desc}", problem_desc)
        
        if "{question}" in output:
            question = random.choice(self.education_knowledge_base[topic]["questions"])
            output = output.replace("{question}", question)
        
        if "{problem}" in output:
            problem = f"A specific application problem of {concept}"
            output = output.replace("{problem}", problem)
        
        if "{duration}" in output:
            duration = f"{random.randint(1, 4)} weeks"
            output = output.replace("{duration}", duration)

        return output
    
    def _adjust_output_by_difficulty(self, output: str, concept: str, difficulty: str) -> str:
        '''
        Adjust the output content based on the difficulty level
        '''
        explanation_style = self.difficulty_templates[difficulty]["explanation_style"]

        if difficulty == "easy":
            # Easy difficulty: Basic explanation, avoid professional terminology
            output = output + f"\n\nSimply put, {concept} is..."
            
            # Add simple examples
            examples = [
                f"\n\nFor example: it's like...",
                f"\n\nFor instance, in daily life...",
                f"\n\nA simple example is..."
            ]
            if random.random() > 0.5:
                output += random.choice(examples)
                
        elif difficulty == "medium":
            # Medium difficulty: Detailed explanation with some professional content
            output = output + f"\n\nFrom a professional perspective, {concept} involves..."
            
            # Add key points
            key_points = [
                f"\n\nKey point 1:...",
                f"\n\nImportant aspect:...",
                f"\n\nSpecial attention should be paid to:..."
            ]
            if random.random() > 0.7:
                output += random.choice(key_points)
                
        else:  # hard
            # Hard difficulty: In-depth analysis with academic discussion
            output = output + f"\n\nIn academic research, recent advances in {concept} include..."
            
            # Add in-depth analysis
            analyses = [
                f"\n\nFrom a theoretical perspective...",
                f"\n\nRelated research shows that...",
                f"\n\nThere are different views in academia..."
            ]
            if random.random() > 0.5:
                output += random.choice(analyses)
        
        return output
    
    def generate_sft_data(self, num_samples: int) -> List[Dict[str, Any]]:
        '''
        Generate specified number of SFT data samples
        '''
        sft_data = []
        for i in range(num_samples):
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} samples...")

            sample = self.generate_single_sample()
            sft_data.append(sample)

        # Convert to HuggingFace Dataset format
        dataset = Dataset.from_list(sft_data)

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

    def save_dataset(self, dataset: Dataset, filepath: str = "alpaca_sft_data.json"):
        '''
        Save the dataset to a JSON file
        '''
        # Create directory if not exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Convert to list of dictionaries
        data_list = dataset.to_list()
        
        # Save as JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"\nDataset saved to: {filepath}")
        
        # Also save as JSONL format (for easier processing)
        jsonl_filepath = filepath.replace('.json', '.jsonl')
        with open(jsonl_filepath, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Also saved as JSONL format: {jsonl_filepath}")

def get_project_root():
    """
    Get the project root directory based on the current file location
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

def main():
    '''
    Generate SFT dataset
    '''
    generator = SFTGenerator()

    # Generate dataset
    num_samples = 5000  # Specify the number of samples to generate
    dataset = generator.generate_sft_data(num_samples)

    # Save dataset
    project_root = get_project_root()
    filepath = os.path.join(project_root, "data", "edu_copilot_sft_data.json")
    generator.save_dataset(dataset, filepath)

    return dataset

if __name__ == "__main__":
    # Generate complete dataset (uncomment to run)
    print("\n=== Generating Complete Dataset ===")
    main()