import json
import random
import re
from typing import List, Dict, Any
from datasets import Dataset
import numpy as np
from collections import Counter
import os

class PreferenceGenerator:
    '''
    Synthetic Preference Data
    '''
    def __init__(self, sft_dataset: Dataset):
        self.sft_dataset = sft_dataset

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
        
        # Enhanced education quality degradation strategies
        self.education_degradation_strategies = [
            self._significant_oversimplify_strategy,
            self._major_factual_errors_strategy,
            self._completely_vague_strategy,
            self._heavily_irrelevant_content_strategy,
            self._severely_disorganize_structure_strategy,
            self._remove_all_educational_value_strategy,
            self._major_misconceptions_strategy,
            self._incomplete_explanation_strategy
        ]

        # Education domain knowledge base
        self.education_domains = {
            "Mathematics": {
                "concepts": ["calculus", "linear algebra", "probability theory", "statistics", "discrete mathematics", "number theory", "geometry"],
                "common_errors": ["calculation errors", "formula misuse", "logical flaws", "definition confusion", "proof errors"],
                "key_terms": ["theorem", "proof", "formula", "derivation", "application", "axiom", "corollary"]
            },
            "Computer Science": {
                "concepts": ["programming", "algorithms", "data structures", "machine learning", "databases", "networking", "operating systems"],
                "common_errors": ["syntax errors", "logical errors", "efficiency issues", "security vulnerabilities", "design flaws"],
                "key_terms": ["code", "algorithm", "complexity", "optimization", "implementation", "protocol", "system"]
            },
            "Physics": {
                "concepts": ["mechanics", "electromagnetism", "thermodynamics", "quantum physics", "relativity", "optics", "nuclear physics"],
                "common_errors": ["unit errors", "formula misuse", "concept confusion", "calculation errors", "theoretical misunderstandings"],
                "key_terms": ["law", "principle", "formula", "experiment", "application", "theory", "phenomenon"]
            },
            "Literature": {
                "concepts": ["narrative structure", "character development", "thematic analysis", "literary devices", "genre studies", "critical theory"],
                "common_errors": ["misinterpretation", "oversimplification", "historical inaccuracy", "context errors", "thematic misunderstandings"],
                "key_terms": ["theme", "symbolism", "metaphor", "plot", "character", "style", "narrative"]
            },
            "History": {
                "concepts": ["historical causation", "periodization", "primary sources", "historiography", "social history", "political history"],
                "common_errors": ["anachronisms", "factual inaccuracies", "oversimplification", "bias", "chronological errors"],
                "key_terms": ["era", "revolution", "civilization", "empire", "movement", "reform", "conflict"]
            }
        }

    def generate_preference_data(self, num_samples: int = 5000) -> Dataset:
        '''
        Generate synthetic preference data
        '''
        if self.sft_dataset is None or len(self.sft_dataset) == 0:
            raise ValueError("SFT dataset is required to generate preference data.")
        
        print(f"Generating {num_samples} synthetic preference data samples...")

        sft_samples = self.sft_dataset.to_list()
        preference_data = []

        for i in range(num_samples):
            if (i+1) % 100 == 0:
                print(f"Generated {i+1} preference samples")
            
            # Randomly select sample from SFT dataset
            base_sample = random.choice(sft_samples)

            # Construct prompt
            prompt = base_sample["instruction"]
            if base_sample.get("input", "").strip():
                prompt += "\n" + base_sample["input"]

            # Use the SFT output as the chosen answer
            chosen_answer = base_sample["output"]

            # Generate lowered quality answer with quality control
            rejected_answer = self._generate_rejected_response_with_quality_control(
                prompt, chosen_answer, base_sample.get("topic")
            )

            # Generate preference reason
            reason, criteria_used = self._generate_preference_reason(
                prompt, chosen_answer, rejected_answer
            )

            # Calculate quality difference
            quality_diff = self._calculate_quality_difference(chosen_answer, rejected_answer)
            
            # Ensure minimum quality difference
            if quality_diff < 0.3:
                # Regenerate rejected answer if difference is too small
                rejected_answer = self._regenerate_rejected_response(chosen_answer, base_sample.get("topic"))
                quality_diff = self._calculate_quality_difference(chosen_answer, rejected_answer)

            preference_sample = {
                "prompt": prompt,
                "chosen": chosen_answer,
                "rejected": rejected_answer,
                "reason": reason,
                "criteria_used": criteria_used,
                "topic": base_sample.get("topic", "Unknown"),
                "difficulty": base_sample.get("difficulty", "medium"),
                "method": "fspo",
                "quality_difference": quality_diff
            }
            
            preference_data.append(preference_sample)

        # Convert to Dataset format
        dataset = Dataset.from_list(preference_data)
        
        # Print statistics
        self._print_generation_stats(dataset)
        
        return dataset
    
    def _generate_rejected_response_with_quality_control(self, prompt: str, chosen_answer: str, topic: str) -> str:
        '''
        Generate a lowered quality answer with quality control to ensure significant difference
        '''
        # Choose strategy that creates maximum quality difference
        high_impact_strategies = [
            self._significant_oversimplify_strategy,
            self._major_factual_errors_strategy,
            self._major_misconceptions_strategy,
            self._incomplete_explanation_strategy
        ]
        
        degradation_strategy = random.choice(high_impact_strategies)
        rejected_response = degradation_strategy(chosen_answer, topic)
        
        return rejected_response
    
    def _regenerate_rejected_response(self, chosen_answer: str, topic: str) -> str:
        '''
        Regenerate rejected response with more aggressive degradation
        '''
        # Use the most aggressive strategies
        aggressive_strategies = [
            self._extreme_oversimplify_strategy,
            self._severe_factual_errors_strategy,
            self._major_misconceptions_strategy
        ]
        
        strategy = random.choice(aggressive_strategies)
        return strategy(chosen_answer, topic)
    
    def _significant_oversimplify_strategy(self, response: str, topic: str) -> str:
        """
        Significant oversimplification: Remove most details, keep only basic idea
        """
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return "I don't really know about this topic."
        
        # Keep only 1 sentence and heavily simplify it
        if len(sentences) > 1:
            # Take first sentence and heavily simplify
            simplified = self._simplify_sentence(sentences[0])
        else:
            simplified = self._simplify_sentence(sentences[0])
        
        # Add overly simplistic conclusion
        simplistic_endings = [
            "That's pretty much it.",
            "It's not that complicated.",
            "You get the idea.",
            "Nothing more to it really."
        ]
        simplified += " " + random.choice(simplistic_endings)
        
        return simplified
    
    def _extreme_oversimplify_strategy(self, response: str, topic: str) -> str:
        """
        Extreme oversimplification: Reduce to absolute minimum
        """
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return "Not sure."
        
        # Extract the main subject and create a very simple statement
        first_sentence = sentences[0]
        words = first_sentence.split()
        
        # Create an extremely simplified version
        if len(words) > 5:
            simplified = " ".join(words[:4]) + " is important."
        else:
            simplified = first_sentence
        
        return simplified + " That's all I know."
    
    def _simplify_sentence(self, sentence: str) -> str:
        """Heavily simplify a single sentence"""
        # Remove technical terms and complex phrases
        simplifications = [
            (r"is a branch of", "is part of"),
            (r"enables computers to", "lets computers"),
            (r"without explicit programming", "automatically"),
            (r"Main types include", "There are types like"),
            (r"refers to", "means"),
            (r"involves", "has"),
            (r"typically occurs when", "happens when"),
            (r"fundamentally changes", "changes"),
            (r"comprehensive analysis of", "look at"),
            (r"theoretical framework", "ideas"),
            (r"practical applications", "uses"),
            (r"historical significance", "importance"),
            (r"complex interplay of factors", "many reasons"),
            (r"sophisticated literary analysis", "book analysis")
        ]
        
        simplified = sentence
        for complex, simple in simplifications:
            simplified = re.sub(complex, simple, simplified, flags=re.IGNORECASE)
        
        # Shorten the sentence significantly
        words = simplified.split()
        if len(words) > 10:
            simplified = ' '.join(words[:8]) + "..."
        
        return simplified
    
    def _major_factual_errors_strategy(self, response: str, topic: str) -> str:
        """
        Add major factual errors that significantly change meaning
        """
        major_errors = {
            "Computer Science": [
                ("machine learning", "magic learning"),
                ("algorithm", "random method"),
                ("programming", "typing code"),
                ("data structure", "data container"),
                ("artificial intelligence", "robot thinking"),
                ("database", "information storage"),
                ("function", "piece of code")
            ],
            "Mathematics": [
                ("calculus", "advanced counting"),
                ("theorem", "math rule"),
                ("proof", "explanation"),
                ("equation", "math sentence"),
                ("probability", "chance guess"),
                ("statistics", "number analysis"),
                ("geometry", "shape math")
            ],
            "Physics": [
                ("energy", "power"),
                ("force", "push or pull"),
                ("gravity", "what makes things fall"),
                ("relativity", "everything is relative"),
                ("quantum", "tiny particles"),
                ("thermodynamics", "heat movement"),
                ("electromagnetism", "electricity and magnets")
            ],
            "Literature": [
                ("theme", "main story"),
                ("symbolism", "hidden stuff"),
                ("character development", "how people change"),
                ("narrative", "telling"),
                ("metaphor", "word picture"),
                ("plot", "story events"),
                ("setting", "where it happens")
            ],
            "History": [
                ("revolution", "big fight"),
                ("civilization", "old society"),
                ("empire", "big country"),
                ("renaissance", "art time"),
                ("colonialism", "taking land"),
                ("democracy", "people voting"),
                ("monarchy", "king rule")
            ]
        }
        
        domain = topic if topic in major_errors else "Computer Science"
        errors = major_errors.get(domain, major_errors["Computer Science"])
        
        modified_response = response
        
        # Replace 2-4 key terms with major errors
        num_errors = random.randint(2, 4)
        applied_errors = 0
        
        for correct, error in random.sample(errors, min(num_errors, len(errors))):
            if correct in modified_response.lower():
                modified_response = re.sub(
                    re.escape(correct), error, modified_response, flags=re.IGNORECASE, count=1
                )
                applied_errors += 1
        
        # If no errors were applied, add a generic wrong statement
        if applied_errors == 0:
            wrong_statements = [
                "Actually, many experts disagree with this approach.",
                "This is outdated information from older textbooks.",
                "Recent studies have shown this to be incorrect.",
                "There's a common misconception about this topic."
            ]
            modified_response = random.choice(wrong_statements)
        
        return modified_response
    
    def _severe_factual_errors_strategy(self, response: str, topic: str) -> str:
        """
        Even more severe factual errors strategy
        """
        # Start with major errors
        erroneous_response = self._major_factual_errors_strategy(response, topic)
        
        # Add additional confusion
        confusing_phrases = [
            "I'm not really sure about this though.",
            "This might be completely wrong.",
            "Someone told me this once but I don't remember well.",
            "I think I might be mixing this up with something else."
        ]
        
        return erroneous_response + " " + random.choice(confusing_phrases)
    
    def _completely_vague_strategy(self, response: str, topic: str) -> str:
        """
        Make the answer completely vague and non-committal
        """
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return "It's hard to say exactly, but generally speaking things vary."
        
        # Keep only 1-2 sentences and make them extremely vague
        vague_sentences = []
        for i, sentence in enumerate(sentences[:2]):  # Only first 2 sentences
            vague_prefixes = [
                "I think maybe", "Perhaps", "It could be that", "Some suggest",
                "Generally speaking", "In many cases", "Often", "Typically"
            ]
            vague_sentence = random.choice(vague_prefixes) + " " + sentence.lower()
            vague_sentences.append(vague_sentence)
        
        vague_response = '. '.join(vague_sentences) + '.'
        
        # Add multiple vague disclaimers
        vague_disclaimers = [
            "But this really depends on the specific situation.",
            "Of course, there are many different opinions about this.",
            "This is just one way to look at it though.",
            "Different people might see this differently."
        ]
        vague_response += " " + random.choice(vague_disclaimers)
        
        return vague_response
    
    def _heavily_irrelevant_content_strategy(self, response: str, topic: str) -> str:
        """
        Add heavily irrelevant content that distracts from the main topic
        """
        irrelevant_contents = {
            "Computer Science": [
                "When programming, it's important to take breaks every hour to avoid eye strain.",
                "Many programmers prefer using dark mode in their code editors.",
                "Team communication is crucial in software development projects.",
                "Learning to type faster can significantly improve coding efficiency."
            ],
            "Mathematics": [
                "Math textbooks have gotten much more colorful in recent years.",
                "Using a good calculator can save time on complex calculations.",
                "Many students find math easier to understand with visual aids.",
                "Practice is important, but don't forget to take breaks."
            ],
            "Physics": [
                "Physics labs usually require safety goggles and proper equipment.",
                "Many famous physicists had interesting personal lives and hobbies.",
                "Science documentaries can provide good visual explanations.",
                "The Nobel Prize is a major recognition in the physics community."
            ],
            "Literature": [
                "Many authors write their best work in the morning with coffee.",
                "Book clubs can provide interesting discussions about literature.",
                "Different editions of books may have different cover designs.",
                "Reading in a comfortable chair improves the experience."
            ],
            "History": [
                "Historical movies often take creative liberties with facts.",
                "Museums provide excellent opportunities to see historical artifacts.",
                "Many history books include timelines and maps for reference.",
                "Documentaries can make historical events more engaging."
            ]
        }
        
        domain = topic if topic in irrelevant_contents else "Computer Science"
        irrelevant_options = irrelevant_contents.get(domain, irrelevant_contents["Computer Science"])
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            # If response is short, replace it with irrelevant content
            return random.choice(irrelevant_options)
        
        # Replace most sentences with irrelevant content
        num_replacements = min(2, len(sentences) - 1)
        for i in range(num_replacements):
            if len(sentences) > 1:
                replace_index = random.randint(1, len(sentences) - 1)
                sentences[replace_index] = random.choice(irrelevant_options)
        
        return '. '.join(sentences) + '.'
    
    def _severely_disorganize_structure_strategy(self, response: str, topic: str) -> str:
        """
        Severely disorganize the structure to make it confusing
        """
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return response  # Too short to effectively disorganize
        
        # Completely shuffle all sentences (don't keep first sentence)
        random.shuffle(sentences)
        
        # Remove all logical connectors
        logic_connectors = ["First", "Then", "Next", "Finally", "Therefore", "So", "However", 
                          "Additionally", "Moreover", "Furthermore", "Consequently", "Because",
                          "Since", "Although", "While", "Whereas"]
        
        disorganized_sentences = []
        for sentence in sentences:
            for connector in logic_connectors:
                if sentence.startswith(connector):
                    sentence = sentence[len(connector):].strip()
                    break
            disorganized_sentences.append(sentence)
        
        # Add confusing transitions
        confusing_transitions = ["Also,", "Anyway,", "So like,", "You know,", "I mean,"]
        for i in range(min(2, len(disorganized_sentences))):
            if random.random() > 0.5:
                disorganized_sentences[i] = random.choice(confusing_transitions) + " " + disorganized_sentences[i]
        
        disorganized_response = '. '.join(disorganized_sentences) + '.'
        
        return disorganized_response
    
    def _remove_all_educational_value_strategy(self, response: str, topic: str) -> str:
        """
        Remove all educational value, leaving only superficial comments
        """
        # Extract key sentences and replace with superficial versions
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return "This topic is interesting but complicated."
        
        # Keep only 1 sentence and make it superficial
        first_sentence = sentences[0]
        
        # Remove educational content
        educational_phrases = [
            "educational significance", "learning value", "important insights", "key points",
            "teaching methods", "learning strategies", "depth of understanding", "knowledge system",
            "critical thinking", "analytical skills", "conceptual framework", "theoretical basis"
        ]
        
        for phrase in educational_phrases:
            if phrase in first_sentence:
                # Replace with simple statement
                first_sentence = "This is about " + topic.lower() + "."
                break
        
        # Simplify complex explanations
        simplification_patterns = [
            (r"Specifically[^\.]+\.", "."),
            (r"Importantly[^\.]+\.", "."),
            (r"The key point is[^\.]+\.", "."),
            (r"Fundamentally[^\.]+\.", "."),
            (r"Essentially[^\.]+\.", ".")
        ]
        
        for pattern, replacement in simplification_patterns:
            first_sentence = re.sub(pattern, replacement, first_sentence)
        
        superficial_response = first_sentence
        
        # Add non-educational conclusion
        non_educational_endings = [
            "But you can look it up if you want more details.",
            "That's just my basic understanding of it.",
            "I'm sure there are better explanations available.",
            "It's one of those topics that experts debate about."
        ]
        superficial_response += " " + random.choice(non_educational_endings)
        
        return superficial_response
    
    def _major_misconceptions_strategy(self, response: str, topic: str) -> str:
        """
        Introduce major misconceptions about the topic
        """
        misconceptions = {
            "Computer Science": [
                "Programming is mostly about memorizing syntax.",
                "Algorithms are just step-by-step instructions for computers.",
                "Machine learning is basically pattern recognition.",
                "You need to be good at math to learn programming."
            ],
            "Mathematics": [
                "Math is all about calculations and formulas.",
                "The goal of math is to get the right answer.",
                "Advanced math is only for scientists and engineers.",
                "You're either born good at math or you're not."
            ],
            "Physics": [
                "Physics is just the study of how things move.",
                "The main goal of physics is to explain natural phenomena.",
                "Physics theories are proven facts about the universe.",
                "You need expensive equipment to do physics research."
            ],
            "Literature": [
                "Literature analysis is just finding hidden meanings.",
                "The author's intention is the only correct interpretation.",
                "Classic literature is always better than modern works.",
                "Good writing follows strict rules and formulas."
            ],
            "History": [
                "History is just memorizing dates and events.",
                "The winner's perspective is the true historical record.",
                "Historical figures were either completely good or bad.",
                "History repeats itself in predictable cycles."
            ]
        }
        
        domain = topic if topic in misconceptions else "Computer Science"
        misconception_options = misconceptions.get(domain, misconceptions["Computer Science"])
        
        # Start with a major misconception
        misconception_response = random.choice(misconception_options)
        
        return misconception_response
    
    def _incomplete_explanation_strategy(self, response: str, topic: str) -> str:
        """
        Provide an incomplete explanation that misses key points
        """
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return response + " But there's more to it that I can't explain right now."
        
        # Remove 70-90% of the content
        keep_percentage = random.uniform(0.1, 0.3)  # Keep only 10-30%
        keep_count = max(1, int(len(sentences) * keep_percentage))
        keep_indices = sorted(random.sample(range(len(sentences)), keep_count))
        
        kept_sentences = [sentences[i] for i in keep_indices]
        incomplete_response = '. '.join(kept_sentences) + '.'
        
        # Add acknowledgment of incompleteness
        incomplete_phrases = [
            "There are other aspects but this covers the basics.",
            "This is a simplified version of a more complex topic.",
            "I'm skipping some details for brevity.",
            "The full explanation is more involved than this."
        ]
        incomplete_response += " " + random.choice(incomplete_phrases)
        
        return incomplete_response
    
    def _generate_preference_reason(self, prompt: str, chosen: str, rejected: str) -> tuple:
        """
        Generate preference reason
        Returns reason and criteria used
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
        Enhanced to be more sensitive to quality differences
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
    
    def save_preference_data(self, dataset: Dataset, filepath: str = "fspo_preference_data.json"):
        """Save preference data"""
        # Create directory if not exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

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
        
        print(f"Preference data saved to: {filepath}")
        print(f"Also saved as JSONL format: {jsonl_filepath}")

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

def main():
    """Main function: Generate preference data using FSPO method"""
    
    # Load SFT data from the specified path
    sft_file_path = "data/edu_copilot_sft_data.json"
    
    try:
        sft_dataset = load_sft_data(sft_file_path)
        
        # Initialize generator with the loaded SFT data
        generator = PreferenceGenerator(sft_dataset=sft_dataset)
        
        # Determine how many samples to generate
        num_sft_samples = len(sft_dataset)
        num_preference_samples = num_sft_samples
        
        print(f"SFT dataset contains {num_sft_samples} samples")
        print(f"Generating {num_preference_samples} preference samples...")
        
        # Generate preference data
        preference_dataset = generator.generate_preference_data(num_preference_samples)
        
        # Save data
        output_filepath = "data/edu_copilot_preference_data.json"
        generator.save_preference_data(preference_dataset, output_filepath)
        
        print("Preference data generation completed successfully!")
        
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
    main()