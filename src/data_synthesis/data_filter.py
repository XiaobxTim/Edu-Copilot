import json
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
from pathlib import Path

class PairedEducationalDataFilter:
    def __init__(self, target_count=3000):
        self.target_count = target_count
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Educational relevance keywords
        self.educational_keywords = {
            'general': ['learn', 'teach', 'study', 'education', 'knowledge', 'skill', 
                       'explain', 'understand', 'analyze', 'discuss', 'evaluate',
                       'concept', 'principle', 'method', 'technique', 'approach'],
            'math': ['calculate', 'solve', 'equation', 'theorem', 'proof', 'formula',
                    'derivative', 'integral', 'matrix', 'probability', 'statistics'],
            'computer_science': ['program', 'algorithm', 'code', 'data structure', 'system', 'network',
                               'software', 'hardware', 'database', 'machine learning', 'ai'],
            'physics': ['force', 'energy', 'motion', 'theory', 'experiment', 'law',
                       'quantum', 'relativity', 'thermodynamics', 'electromagnetism'],
            'literature': ['analyze', 'theme', 'character', 'plot', 'symbolism', 'narrative',
                         'poetry', 'novel', 'drama', 'criticism', 'interpretation'],
            'history': ['event', 'period', 'civilization', 'revolution', 'culture', 'historical',
                       'ancient', 'modern', 'political', 'economic', 'social']
        }
        
        # Filtering parameters
        self.min_length = 20
        self.max_length = 2000
        self.min_quality_difference = 0.1
        self.blacklist_keywords = ['test', 'example', 'dummy', 'placeholder']

    def paired_filtering_pipeline(self, sft_data, preference_data):
        """Paired filtering pipeline that maintains correspondence and ensures exact 3000 samples"""
        print(f"Original data - SFT: {len(sft_data)}, Preference: {len(preference_data)}")
        
        # Step 1: Create paired data
        paired_data = self._create_paired_data(sft_data, preference_data)
        print(f"Created {len(paired_data)} data pairs")
        
        # Step 2: Calculate comprehensive quality scores and rank
        scored_pairs = self._score_and_rank_pairs(paired_data)
        print(f"Scored and ranked {len(scored_pairs)} pairs")
        
        # Step 3: Select top 3000 pairs
        selected_pairs = scored_pairs[:self.target_count]
        print(f"Selected top {len(selected_pairs)} pairs")
        
        # Step 4: Separate the data
        final_sft = [pair['sft'] for pair in selected_pairs]
        final_pref = [pair['preference'] for pair in selected_pairs]
        
        # Validate results
        self._validate_results(final_sft, final_pref)
        
        return final_sft, final_pref

    def _create_paired_data(self, sft_data, preference_data):
        """Create paired SFT and preference data"""
        paired_data = []
        
        # Assume data is sequentially aligned
        min_len = min(len(sft_data), len(preference_data))
        
        for i in range(min_len):
            paired_data.append({
                'sft': sft_data[i],
                'preference': preference_data[i],
                'pair_id': i
            })
        
        return paired_data

    def _score_and_rank_pairs(self, paired_data):
        """Calculate comprehensive quality scores and rank pairs"""
        scored_pairs = []
        
        for pair in paired_data:
            score = self._calculate_comprehensive_score(pair)
            scored_pairs.append((pair, score))
        
        # Sort by score in descending order
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [pair for pair, score in scored_pairs]

    def _calculate_comprehensive_score(self, pair):
        """Calculate comprehensive quality score (combining four filtering dimensions)"""
        sft_sample = pair['sft']
        pref_sample = pair['preference']
        
        # Dimension 1: Basic Quality Score (30%)
        quality_score = self._calculate_quality_score(sft_sample, pref_sample) * 0.3
        
        # Dimension 2: Educational Relevance Score (25%)
        edu_score = self._calculate_educational_relevance_score(sft_sample, pref_sample) * 0.25
        
        # Dimension 3: Diversity Score (25%)
        diversity_score = self._calculate_diversity_score(sft_sample) * 0.25
        
        # Dimension 4: Difficulty Balance Score (20%)
        difficulty_score = self._calculate_difficulty_score(sft_sample) * 0.2
        
        total_score = quality_score + edu_score + diversity_score + difficulty_score
        
        return total_score

    def _calculate_quality_score(self, sft_sample, pref_sample):
        """Calculate basic quality score"""
        score = 0.0
        
        # SFT sample quality check
        sft_quality = self._check_sft_quality(sft_sample)
        pref_quality = self._check_preference_quality(pref_sample)
        
        if sft_quality and pref_quality:
            score += 0.6  # Basic quality passed
            
            # Text length score
            sft_text = sft_sample.get('instruction', '') + sft_sample.get('input', '') + sft_sample.get('output', '')
            pref_text = pref_sample.get('prompt', '') + pref_sample.get('chosen', '') + pref_sample.get('rejected', '')
            
            if 50 <= len(sft_text) <= 1000:
                score += 0.2
            if 50 <= len(pref_text) <= 1000:
                score += 0.2
        
        return min(score, 1.0)

    def _check_sft_quality(self, sample):
        """Check SFT sample quality"""
        if not isinstance(sample, dict):
            return False
        
        required_fields = ['instruction', 'output']
        for field in required_fields:
            if field not in sample or not sample[field].strip():
                return False
        
        text = sample.get('instruction', '') + sample.get('input', '') + sample.get('output', '')
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.blacklist_keywords):
            return False
        
        return True

    def _check_preference_quality(self, sample):
        """Check preference sample quality"""
        if not isinstance(sample, dict):
            return False
        
        required_fields = ['prompt', 'chosen', 'rejected']
        for field in required_fields:
            if field not in sample or not sample[field].strip():
                return False
        
        quality_diff = sample.get('quality_difference', 0)
        if quality_diff < self.min_quality_difference:
            return False
        
        text = sample['prompt'] + sample['chosen'] + sample['rejected']
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.blacklist_keywords):
            return False
        
        return True

    def _calculate_educational_relevance_score(self, sft_sample, pref_sample):
        """Calculate educational relevance score"""
        score = 0.0
        
        # Combine texts for analysis
        sft_text = sft_sample.get('instruction', '') + sft_sample.get('input', '') + sft_sample.get('output', '')
        pref_text = pref_sample.get('prompt', '') + pref_sample.get('chosen', '') + pref_sample.get('rejected', '')
        combined_text = sft_text + ' ' + pref_text
        
        text_lower = combined_text.lower()
        
        # Calculate educational keyword coverage
        keyword_count = 0
        for category_keywords in self.educational_keywords.values():
            for keyword in category_keywords:
                if keyword in text_lower:
                    keyword_count += 1
                    break
        
        # Keyword coverage score
        if keyword_count >= 3:
            score += 0.6
        elif keyword_count >= 2:
            score += 0.4
        elif keyword_count >= 1:
            score += 0.2
        
        # Topic consistency check
        sft_topic = sft_sample.get('topic', '')
        pref_topic = pref_sample.get('topic', '')
        if sft_topic and pref_topic and sft_topic == pref_topic:
            score += 0.4
        
        return min(score, 1.0)

    def _calculate_diversity_score(self, sft_sample):
        """Calculate diversity score (based on content uniqueness)"""
        # Use text length and content features as diversity proxies
        text = sft_sample.get('instruction', '') + sft_sample.get('input', '') + sft_sample.get('output', '')
        
        # Length diversity
        length = len(text)
        if 100 <= length <= 500:  # Ideal length range
            length_score = 0.4
        else:
            length_score = 0.2
        
        # Content richness (sentence count)
        sentences = [s for s in text.split('.') if s.strip()]
        sentence_count = len(sentences)
        if sentence_count >= 3:
            richness_score = 0.3
        elif sentence_count >= 2:
            richness_score = 0.2
        else:
            richness_score = 0.1
        
        # Vocabulary diversity (simple implementation)
        words = text.split()
        unique_words = set(words)
        if len(words) > 0:
            diversity_ratio = len(unique_words) / len(words)
            vocab_score = diversity_ratio * 0.3
        else:
            vocab_score = 0
        
        return length_score + richness_score + vocab_score

    def _calculate_difficulty_score(self, sft_sample):
        """Calculate difficulty balance score"""
        difficulty = sft_sample.get('difficulty', 'medium')
        
        # Difficulty distribution score (encourage diversity)
        if difficulty == 'easy':
            return 0.3  # Encourage more medium and hard samples
        elif difficulty == 'medium':
            return 0.5  # Medium difficulty is most ideal
        else:  # hard
            return 0.2  # Hard samples also have value

    def _validate_results(self, sft_data, preference_data):
        """Validate filtering results"""
        print(f"\n=== Validation Results ===")
        print(f"SFT samples: {len(sft_data)}")
        print(f"Preference samples: {len(preference_data)}")
        
        if len(sft_data) != len(preference_data):
            print("ERROR: SFT and Preference counts don't match!")
            return False
        
        if len(sft_data) != self.target_count:
            print(f"ERROR: Expected {self.target_count} samples, got {len(sft_data)}")
            return False
        
        # Check data quality
        quality_scores = []
        for sft, pref in zip(sft_data, preference_data):
            score = self._calculate_comprehensive_score({'sft': sft, 'preference': pref})
            quality_scores.append(score)
        
        print(f"Average quality score: {np.mean(quality_scores):.3f}")
        print(f"Min quality score: {min(quality_scores):.3f}")
        print(f"Max quality score: {max(quality_scores):.3f}")
        
        # Topic distribution
        topics = [s.get('topic', 'Unknown') for s in sft_data]
        topic_counts = Counter(topics)
        print(f"\nTopic distribution:")
        for topic, count in topic_counts.most_common():
            print(f"  {topic}: {count} samples ({count/len(sft_data)*100:.1f}%)")
        
        # Difficulty distribution
        difficulties = [s.get('difficulty', 'medium') for s in sft_data]
        diff_counts = Counter(difficulties)
        print(f"Difficulty distribution:")
        for diff, count in diff_counts.most_common():
            print(f"  {diff}: {count} samples ({count/len(sft_data)*100:.1f}%)")
        
        return True

    def save_filtered_data(self, sft_data, preference_data, sft_path, pref_path):
        """Save filtered data"""
        with open(sft_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)
        
        with open(pref_path, 'w', encoding='utf-8') as f:
            json.dump(preference_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nFiltered data saved:")
        print(f"SFT data: {sft_path} ({len(sft_data)} samples)")
        print(f"Preference data: {pref_path} ({len(preference_data)} samples)")

    def analyze_dataset_quality(self, data, data_type="SFT"):
        """Analyze dataset quality metrics"""
        print(f"\n=== {data_type} Data Quality Analysis ===")
        
        if len(data) == 0:
            print("No data to analyze")
            return
        
        # Text length analysis
        lengths = []
        for sample in data:
            if data_type == "SFT":
                text = sample.get('instruction', '') + ' ' + sample.get('input', '') + ' ' + sample.get('output', '')
            else:
                text = sample['prompt'] + ' ' + sample['chosen'] + ' ' + sample['rejected']
            lengths.append(len(text))
        
        print(f"Average text length: {np.mean(lengths):.1f} characters")
        print(f"Min length: {min(lengths)}, Max length: {max(lengths)}")
        
        # Topic distribution
        topics = [s.get('topic', 'Unknown') for s in data]
        print(f"\nTopic distribution:")
        for topic, count in Counter(topics).most_common():
            print(f"  {topic}: {count} ({count/len(data)*100:.1f}%)")
        
        # Difficulty distribution
        difficulties = [s.get('difficulty', 'medium') for s in data]
        print(f"\nDifficulty distribution:")
        for diff, count in Counter(difficulties).most_common():
            print(f"  {diff}: {count} ({count/len(data)*100:.1f}%)")
        
        # Quality difference analysis for preference data
        if data_type == "Preference":
            quality_diffs = [s.get('quality_difference', 0) for s in data]
            print(f"\nQuality difference analysis:")
            print(f"  Average: {np.mean(quality_diffs):.3f}")
            print(f"  Std: {np.std(quality_diffs):.3f}")
            print(f"  Min: {min(quality_diffs):.3f}, Max: {max(quality_diffs):.3f}")

# Main execution function
def main():
    DIR = Path(__file__).parent.resolve()
    SFT_PATH = DIR.parent.parent / 'data' / 'edu_copilot_sft_data.json'
    PREF_PATH = DIR.parent.parent / 'data' / 'edu_copilot_preference_data.json'
    # Load your datasets
    try:
        with open(SFT_PATH, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)
        
        with open(PREF_PATH, 'r', encoding='utf-8') as f:
            preference_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return
    
    # Initialize filter
    filter = PairedEducationalDataFilter(target_count=3000)
    
    # Analyze original data
    print("=== Original Dataset Analysis ===")
    filter.analyze_dataset_quality(sft_data, "SFT")
    filter.analyze_dataset_quality(preference_data, "Preference")
    
    # Execute filtering pipeline
    print("\n=== Starting Paired Filtering Pipeline ===")
    filtered_sft, filtered_pref = filter.paired_filtering_pipeline(sft_data, preference_data)

    Filter_DIR = DIR.parent.parent / 'data'
    
    # Save results
    filter.save_filtered_data(
        filtered_sft, 
        filtered_pref, 
        Filter_DIR / 'filtered_sft_data.json', 
        Filter_DIR / 'filtered_preference_data.json'
    )
    
    print("\n=== Filtering Complete ===")
    print(f"Successfully filtered to exactly {len(filtered_sft)} paired samples")
    print(f"SFT: {len(filtered_sft)} samples, Preference: {len(filtered_pref)} samples")

if __name__ == "__main__":
    main()