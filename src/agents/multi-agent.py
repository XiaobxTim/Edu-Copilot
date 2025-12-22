import json
import random
from typing import Dict, List, Any
from openai import OpenAI
import os
import argparse

class EducationalAgent:
    """Base class for educational AI agents"""
    def __init__(self, name: str, system_prompt: str, model: str = "deepseek-chat"):
        self.name = name
        # Initialize OpenAI client for DeepSeek API
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model
        self.system_prompt = system_prompt

    def call(self, user_message: str, temperature: float = 0.7) -> str:
        """Call LLM to generate response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error in {self.name}: {str(e)}"

class ContentGeneratorAgent(EducationalAgent):
    """Educational content generation agent"""
    def __init__(self):
        system_prompt = """You are a professional educational content creation expert, skilled at developing high-quality learning materials and practice questions based on subject, topic, and difficulty level.
        Your materials should be accurate, easy to understand, and educationally valuable, effectively helping students master relevant knowledge."""
        super().__init__("ContentGenerator", system_prompt)

    def generate_material(self, subject: str, topic: str, difficulty: str) -> str:
        """Generate learning materials"""
        prompt = f"""Please generate learning materials for the topic "{topic}" in the subject {subject}, with difficulty level {difficulty}.
        The content should include:
        1. Explanation of core concepts
        2. Key knowledge points
        3. Practical application cases
        4. 3 practice questions (including answers)"""
        return self.call(prompt)

class AssessmentAgent(EducationalAgent):
    """Learning assessment agent"""
    def __init__(self):
        system_prompt = """You are a strict and fair educational assessment expert, skilled at grading student assignments and providing constructive feedback.
        Your feedback should be specific, targeted, and include improvement suggestions to help students understand their mistakes and make progress."""
        super().__init__("AssessmentAgent", system_prompt)

    def evaluate(self, subject: str, question: str, student_answer: str, reference_answer: str) -> str:
        """Evaluate student's answer"""
        prompt = f"""Please evaluate the following answer for the subject {subject}:
        Question: {question}
        Student's Answer: {student_answer}
        Reference Answer: {reference_answer}
        
        The evaluation should include:
        1. Accuracy score (1-10 points)
        2. Error analysis (if applicable)
        3. Improvement suggestions
        4. Tips for consolidating relevant knowledge points"""
        return self.call(prompt)

class ConceptExplainerAgent(EducationalAgent):
    """Concept explanation agent"""
    def __init__(self):
        system_prompt = """You are an educational expert skilled at explaining complex concepts, able to adjust the depth and method of explanation according to the student's level of understanding.
        Your explanations should be clear, vivid, and good at using analogies and examples to help students truly understand the essence of the concept."""
        super().__init__("ConceptExplainer", system_prompt)

    def explain(self, concept: str, student_level: str, confusion_point: str = "") -> str:
        """Explain a concept"""
        prompt = f"""Please explain the concept "{concept}" to a student at the {student_level} level.
        {'The student is confused about: ' + confusion_point if confusion_point else ''}
        The explanation should:
        1. Be concise and easy to understand, avoiding unnecessary professional jargon
        2. Include analogies or examples from daily life
        3. Have a clear structure and logical coherence
        4. Clarify possible misunderstandings of the student"""
        return self.call(prompt, temperature=0.8)

class LearningPathAgent(EducationalAgent):
    """Learning path planning agent"""
    def __init__(self):
        system_prompt = """You are an experienced educational planning expert, skilled at developing personalized learning paths based on the student's current level, learning goals, and time schedule.
        Your plans should be scientific, reasonable, and progressive, effectively helping students achieve their learning goals."""
        super().__init__("LearningPathAgent", system_prompt)

    def create_plan(self, subject: str, current_level: str, target: str, time_available: str) -> str:
        """Create a learning plan"""
        prompt = f"""Please develop a learning plan for the student in the subject {subject}:
        Current Level: {current_level}
        Learning Goal: {target}
        Available Time: {time_available}
        
        The learning plan should include:
        1. Division of phased goals
        2. Weekly learning content arrangement
        3. Key assessment nodes
        4. Recommended learning resources
        5. Learning method suggestions"""
        return self.call(prompt, temperature=0.6)

class EducationalCopilot:
    """Coordinator for the educational multi-agent system"""
    def __init__(self):
        self.content_generator = ContentGeneratorAgent()
        self.assessor = AssessmentAgent()
        self.explainer = ConceptExplainerAgent()
        self.path_planner = LearningPathAgent()
        self.student_profile = {}

    def set_student_profile(self, profile: Dict[str, Any]):
        """Set student profile"""
        self.student_profile = profile

    def get_learning_plan(self, subject: str, target: str, time_available: str) -> str:
        """Get learning plan"""
        current_level = self.student_profile.get("current_level", "beginner")
        return self.path_planner.create_plan(subject, current_level, target, time_available)

    def get_learning_material(self, subject: str, topic: str) -> str:
        """Get learning materials"""
        difficulty = self._determine_difficulty()
        return self.content_generator.generate_material(subject, topic, difficulty)

    def submit_assignment(self, subject: str, question: str, student_answer: str, reference_answer: str) -> str:
        """Submit assignment and get evaluation"""
        assessment = self.assessor.evaluate(subject, question, student_answer, reference_answer)
        # Update student profile based on assessment results
        self._update_student_profile_based_on_assessment(assessment)
        return assessment

    def get_concept_explanation(self, concept: str, confusion_point: str = "") -> str:
        """Get concept explanation"""
        student_level = self.student_profile.get("current_level", "beginner")
        return self.explainer.explain(concept, student_level, confusion_point)

    def _determine_difficulty(self) -> str:
        """Determine difficulty based on student level"""
        level = self.student_profile.get("current_level", "beginner")
        level_map = {
            "beginner": "easy",
            "intermediate": "medium",
            "advanced": "hard"
        }
        return level_map.get(level, "medium")

    def _update_student_profile_based_on_assessment(self, assessment: str):
        """Update student profile based on assessment results"""
        # Simple score extraction logic; more complex NLP parsing can be used in practical applications
        if "Score: 10 points" in assessment:
            self.student_profile["mastered_topics"] = self.student_profile.get("mastered_topics", []) + [self.student_profile.get("current_topic")]
        elif "Score: below 5 points" in assessment:
            self.student_profile["weak_topics"] = self.student_profile.get("weak_topics", []) + [self.student_profile.get("current_topic")]

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Educational Multi-Agent System (EducationalCopilot)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for learning plan
    plan_parser = subparsers.add_parser("plan", help="Generate a personalized learning plan")
    plan_parser.add_argument("--subject", required=True, type=str, help="Subject of the learning plan (e.g., Computer Science)")
    plan_parser.add_argument("--target", required=True, type=str, help="Learning goal (e.g., Master Python Basic Syntax)")
    plan_parser.add_argument("--time-available", required=True, type=str, help="Available time for learning (e.g., 5 hours per week for 4 weeks)")
    plan_parser.add_argument("--current-level", default="beginner", type=str, choices=["beginner", "intermediate", "advanced"], help="Student's current level (default: beginner)")
    plan_parser.add_argument("--current-topic", type=str, help="Student's current learning topic (e.g., Python Basic Syntax)")

    # Subparser for learning material
    material_parser = subparsers.add_parser("material", help="Generate learning materials for a specific topic")
    material_parser.add_argument("--subject", required=True, type=str, help="Subject of the material (e.g., Computer Science)")
    material_parser.add_argument("--topic", required=True, type=str, help="Specific topic (e.g., Python Functions)")
    material_parser.add_argument("--current-level", default="beginner", type=str, choices=["beginner", "intermediate", "advanced"], help="Student's current level (default: beginner)")
    material_parser.add_argument("--current-topic", type=str, help="Student's current learning topic (e.g., Python Functions)")

    # Subparser for assignment evaluation
    assess_parser = subparsers.add_parser("assess", help="Evaluate student's assignment answer")
    assess_parser.add_argument("--subject", required=True, type=str, help="Subject of the assignment (e.g., Computer Science)")
    assess_parser.add_argument("--question", required=True, type=str, help="The assignment question (e.g., Write a Python function to calculate the average of a list)")
    assess_parser.add_argument("--student-answer", required=True, type=str, help="Student's answer to the question")
    assess_parser.add_argument("--reference-answer", required=True, type=str, help="Reference answer for the question")
    assess_parser.add_argument("--current-level", default="beginner", type=str, choices=["beginner", "intermediate", "advanced"], help="Student's current level (default: beginner)")
    assess_parser.add_argument("--current-topic", type=str, help="Student's current learning topic (e.g., Python Functions)")

    # Subparser for concept explanation
    explain_parser = subparsers.add_parser("explain", help="Get explanation for a specific concept")
    explain_parser.add_argument("--concept", required=True, type=str, help="Concept to explain (e.g., Recursion in Functions)")
    explain_parser.add_argument("--confusion-point", default="", type=str, help="Student's confusion point about the concept (e.g., Don't understand how recursion terminates)")
    explain_parser.add_argument("--current-level", default="beginner", type=str, choices=["beginner", "intermediate", "advanced"], help="Student's current level (default: beginner)")
    explain_parser.add_argument("--current-topic", type=str, help="Student's current learning topic (e.g., Python Functions)")

    # Parse arguments
    args = parser.parse_args()

    # Initialize EducationalCopilot
    copilot = EducationalCopilot()

    # Set student profile based on arguments
    student_profile = {
        "current_level": args.current_level,
        "current_topic": args.current_topic if hasattr(args, "current_topic") and args.current_topic else "",
        "mastered_topics": [],
        "weak_topics": []
    }
    copilot.set_student_profile(student_profile)

    # Execute corresponding command
    if args.command == "plan":
        result = copilot.get_learning_plan(args.subject, args.target, args.time_available)
        print("=== Learning Plan ===")
        print(result)
    elif args.command == "material":
        result = copilot.get_learning_material(args.subject, args.topic)
        print("=== Learning Materials ===")
        print(result)
    elif args.command == "assess":
        result = copilot.submit_assignment(args.subject, args.question, args.student_answer, args.reference_answer)
        print("=== Assignment Evaluation ===")
        print(result)
    elif args.command == "explain":
        result = copilot.get_concept_explanation(args.concept, args.confusion_point)
        print("=== Concept Explanation ===")
        print(result)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()