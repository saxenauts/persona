"""
PersonaMem Dataset Loader

Loads and processes the PersonaMem benchmark dataset for evaluation.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PersonaMemQuestion:
    """Represents a single PersonaMem question"""

    question_id: str
    question_type: str
    question: str
    options: Dict[str, str]  # {'a': 'option text', 'b': ...}
    correct_answer: str  # 'a', 'b', 'c', or 'd'
    context: str  # Conversation history
    metadata: Dict[str, Any]  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question": self.question,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "context": self.context,
            "metadata": self.metadata,
        }


class PersonaMemLoader:
    """Loader for PersonaMem dataset"""

    QUESTION_TYPES = [
        "recall_user_shared_facts",
        "suggest_new_ideas",
        "acknowledge_latest_user_preferences",
        "track_full_preference_evolution",
        "recalling_the_reasons_behind_previous_updates",
        "provide_preference_aligned_recommendations",
        "generalizing_to_new_scenarios",
    ]

    def __init__(self, data_dir: str = "evals/data/personamem", variant: str = "32k"):
        """
        Initialize PersonaMem loader

        Args:
            data_dir: Directory containing PersonaMem data
            variant: Dataset variant ('32k', '128k', or '1M')
        """
        self.data_dir = Path(data_dir)
        self.variant = variant

        # Path to questions file
        self.questions_path = self.data_dir / f"questions_{variant}_{variant}.json"

        if not self.questions_path.exists():
            raise FileNotFoundError(
                f"PersonaMem {variant} data not found at {self.questions_path}. "
                f"Run 'python evals/scripts/download_personamem.py' to download."
            )

    def load(self) -> List[PersonaMemQuestion]:
        """
        Load all questions from the dataset

        Returns:
            List of PersonaMemQuestion objects
        """
        print(f"Loading PersonaMem {self.variant} dataset...")

        with open(self.questions_path, 'r') as f:
            raw_data = json.load(f)

        questions = []
        for idx, item in enumerate(raw_data):
            # Parse the question
            question = self._parse_question(item, idx)
            questions.append(question)

        print(f"Loaded {len(questions)} questions from PersonaMem {self.variant}")
        return questions

    def _parse_question(self, item: Dict[str, Any], idx: int) -> PersonaMemQuestion:
        """
        Parse a raw question item into a PersonaMemQuestion object

        Args:
            item: Raw question data from JSON
            idx: Question index

        Returns:
            PersonaMemQuestion object
        """
        # Extract options
        options = {}
        for key in ['a', 'b', 'c', 'd']:
            option_key = f"option_{key}"
            if option_key in item:
                options[key] = item[option_key]

        # Generate question ID if not present
        question_id = item.get('id', f"personamem_{self.variant}_{idx}")

        # Extract question type
        question_type = item.get('question_type', 'unknown')

        return PersonaMemQuestion(
            question_id=question_id,
            question_type=question_type,
            question=item.get('question', ''),
            options=options,
            correct_answer=item.get('answer', '').lower().strip(),
            context=item.get('context', ''),
            metadata={
                "variant": self.variant,
                "index": idx,
                **{k: v for k, v in item.items() if k not in [
                    'id', 'question_type', 'question', 'option_a', 'option_b',
                    'option_c', 'option_d', 'answer', 'context'
                ]}
            }
        )

    def load_by_type(self, question_type: str) -> List[PersonaMemQuestion]:
        """
        Load questions filtered by type

        Args:
            question_type: Type of question to load

        Returns:
            List of questions of the specified type
        """
        all_questions = self.load()
        return [q for q in all_questions if q.question_type == question_type]

    def get_type_distribution(self) -> Dict[str, int]:
        """
        Get distribution of questions by type

        Returns:
            Dictionary mapping question type to count
        """
        questions = self.load()
        distribution = {}

        for q in questions:
            qtype = q.question_type
            distribution[qtype] = distribution.get(qtype, 0) + 1

        return distribution

    def save_subset(self, questions: List[PersonaMemQuestion], output_path: str):
        """
        Save a subset of questions to file

        Args:
            questions: List of questions to save
            output_path: Path to save to
        """
        data = [q.to_dict() for q in questions]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(questions)} questions to {output_path}")


# Example usage
if __name__ == "__main__":
    loader = PersonaMemLoader(variant="32k")

    # Load all questions
    questions = loader.load()
    print(f"\nTotal questions: {len(questions)}")

    # Get type distribution
    distribution = loader.get_type_distribution()
    print("\nQuestion Type Distribution:")
    for qtype, count in sorted(distribution.items()):
        print(f"  {qtype}: {count}")

    # Load specific type
    fact_recall_questions = loader.load_by_type("recall_user_shared_facts")
    print(f"\nFact recall questions: {len(fact_recall_questions)}")
