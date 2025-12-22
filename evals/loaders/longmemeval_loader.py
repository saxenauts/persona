"""
LongMemEval Dataset Loader

Loads and processes the LongMemEval Oracle dataset for evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LongMemEvalQuestion:
    """Represents a single LongMemEval question"""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_dates: List[str]
    haystack_session_ids: List[str]
    haystack_sessions: List[List[Dict[str, str]]]
    is_abstention: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
            "question_date": self.question_date,
            "haystack_dates": self.haystack_dates,
            "haystack_session_ids": self.haystack_session_ids,
            "haystack_sessions": self.haystack_sessions,
            "is_abstention": self.is_abstention,
            "metadata": self.metadata,
        }


class LongMemEvalLoader:
    """Loader for LongMemEval Oracle dataset"""

    QUESTION_TYPES = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    ]

    def __init__(self, data_path: str = "evals/data/longmemeval_oracle.json"):
        """
        Initialize LongMemEval loader

        Args:
            data_path: Path to LongMemEval Oracle JSON file
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"LongMemEval Oracle data not found at {self.data_path}"
            )

    def load(self) -> List[LongMemEvalQuestion]:
        """
        Load all questions from the dataset

        Returns:
            List of LongMemEvalQuestion objects
        """
        print(f"Loading LongMemEval Oracle dataset from {self.data_path}...")

        with open(self.data_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            if 'data' in data:
                questions_data = data['data']
            else:
                questions_data = list(data.values())[0] if data else []
        else:
            questions_data = data

        questions = []
        for item in questions_data:
            question = self._parse_question(item)
            questions.append(question)

        print(f"Loaded {len(questions)} questions from LongMemEval Oracle")
        return questions

    def _parse_question(self, item: Dict[str, Any]) -> LongMemEvalQuestion:
        """
        Parse a raw question item into a LongMemEvalQuestion object

        Args:
            item: Raw question data from JSON

        Returns:
            LongMemEvalQuestion object
        """
        question_id = item.get('question_id', '')

        # Check if this is an abstention question
        is_abstention = '_abs' in question_id

        return LongMemEvalQuestion(
            question_id=question_id,
            question_type=item.get('question_type', ''),
            question=item.get('question', ''),
            answer=item.get('answer', ''),
            question_date=item.get('question_date', ''),
            haystack_dates=item.get('haystack_dates', []),
            haystack_session_ids=item.get('haystack_session_ids', []),
            haystack_sessions=item.get('haystack_sessions', []),
            is_abstention=is_abstention,
            metadata={
                k: v for k, v in item.items()
                if k not in [
                    'question_id', 'question_type', 'question', 'answer',
                    'question_date', 'haystack_dates', 'haystack_session_ids',
                    'haystack_sessions'
                ]
            }
        )

    def load_by_type(self, question_type: str) -> List[LongMemEvalQuestion]:
        """
        Load questions filtered by type

        Args:
            question_type: Type of question to load

        Returns:
            List of questions of the specified type
        """
        all_questions = self.load()
        return [q for q in all_questions if q.question_type == question_type]

    def load_abstention_questions(self) -> List[LongMemEvalQuestion]:
        """
        Load only abstention questions

        Returns:
            List of abstention questions
        """
        all_questions = self.load()
        return [q for q in all_questions if q.is_abstention]

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

    def get_abstention_distribution(self) -> Dict[str, int]:
        """
        Get distribution of abstention questions by type

        Returns:
            Dictionary mapping question type to abstention count
        """
        questions = self.load()
        distribution = {}

        for q in questions:
            if q.is_abstention:
                qtype = q.question_type
                distribution[qtype] = distribution.get(qtype, 0) + 1

        return distribution

    def save_subset(self, questions: List[LongMemEvalQuestion], output_path: str):
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
    loader = LongMemEvalLoader()

    # Load all questions
    questions = loader.load()
    print(f"\nTotal questions: {len(questions)}")

    # Get type distribution
    distribution = loader.get_type_distribution()
    print("\nQuestion Type Distribution:")
    for qtype, count in sorted(distribution.items()):
        print(f"  {qtype}: {count}")

    # Get abstention distribution
    abs_distribution = loader.get_abstention_distribution()
    print("\nAbstention Question Distribution:")
    for qtype, count in sorted(abs_distribution.items()):
        print(f"  {qtype}: {count}")

    # Load specific type
    multi_session_questions = loader.load_by_type("multi-session")
    print(f"\nMulti-session questions: {len(multi_session_questions)}")

    # Load abstention questions
    abstention_questions = loader.load_abstention_questions()
    print(f"Total abstention questions: {len(abstention_questions)}")
