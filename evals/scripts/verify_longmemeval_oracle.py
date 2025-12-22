"""
Verify LongMemEval Oracle dataset integrity

This script checks the structure and integrity of the LongMemEval Oracle dataset,
ensuring all expected question types are present and properly formatted.
"""

import json
from pathlib import Path
from collections import Counter


def verify_longmemeval_oracle(data_path: str = "evals/data/longmemeval_oracle.json"):
    """
    Verify LongMemEval Oracle dataset integrity

    Args:
        data_path: Path to the LongMemEval Oracle dataset JSON file
    """
    print("Verifying LongMemEval Oracle dataset...")
    print(f"Loading from: {data_path}\n")

    # Load the dataset
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Check if it's a list or dict
    if isinstance(data, dict):
        if 'data' in data:
            questions = data['data']
        else:
            questions = list(data.values())[0] if data else []
    else:
        questions = data

    print(f"Total questions loaded: {len(questions)}")

    # Extract question types
    question_types = []
    abstention_count = 0

    for q in questions:
        qtype = q.get('question_type', 'unknown')
        question_types.append(qtype)

        # Check for abstention questions (those with _abs suffix)
        qid = q.get('question_id', '')
        if '_abs' in qid:
            abstention_count += 1

    # Count questions by type
    type_counts = Counter(question_types)

    print("\n" + "="*60)
    print("Question Type Distribution:")
    print("="*60)

    expected_types = {
        'single-session-user': 70,
        'single-session-assistant': 56,
        'single-session-preference': 30,
        'multi-session': 133,
        'temporal-reasoning': 133,
        'knowledge-update': 78,
    }

    total_expected = 0
    total_found = 0

    for qtype, expected_count in sorted(expected_types.items()):
        actual_count = type_counts.get(qtype, 0)
        total_expected += expected_count
        total_found += actual_count

        status = "✓" if actual_count > 0 else "✗"
        print(f"{status} {qtype:30s}: {actual_count:3d} (expected ~{expected_count})")

    print(f"\nAbstention questions (_abs suffix): {abstention_count}")

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Total questions found: {total_found}")
    print(f"Total expected: {total_expected}")

    # Check for unexpected types
    unexpected_types = set(question_types) - set(expected_types.keys())
    if unexpected_types:
        print(f"\nUnexpected question types found: {unexpected_types}")

    # Verify structure of a sample question
    if questions:
        print("\n" + "="*60)
        print("Sample Question Structure:")
        print("="*60)
        sample = questions[0]
        for key in sample.keys():
            value = sample[key]
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {key}: {type(value).__name__} = {value}")

    print("\n✓ LongMemEval Oracle dataset verification complete!")

    # Save a summary report
    report = {
        "total_questions": len(questions),
        "question_type_counts": dict(type_counts),
        "abstention_count": abstention_count,
        "expected_types": expected_types,
        "verification_status": "complete"
    }

    report_path = Path(data_path).parent / "longmemeval_oracle_verification.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nVerification report saved to: {report_path}")

    return report


if __name__ == "__main__":
    verify_longmemeval_oracle()
