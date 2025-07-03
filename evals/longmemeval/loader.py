from dataclasses import dataclass
from datetime import datetime
from typing import List, Generator, Optional
import json

@dataclass
class Turn:
    """Represents a single turn in a conversation"""
    role: str
    content: str
    iso_timestamp: str
    has_answer: bool = False

@dataclass 
class LongMemInstance:
    """Represents a complete LongMemEval instance"""
    question_id: str
    question: str
    gold_answer: str
    question_date: str
    question_type: str
    sessions: List[List[Turn]]
    haystack_dates: List[str]
    answer_session_ids: Optional[List[str]] = None

def parse_date_to_iso(date_str: str) -> str:
    """Convert LongMemEval date format to ISO timestamp"""
    try:
        # Handle format: "2023/11/15 (Wed) 14:30"
        dt = datetime.strptime(date_str, "%Y/%m/%d (%a) %H:%M")
        return dt.isoformat()
    except ValueError:
        # Fallback for any date parsing issues
        print(f"Warning: Could not parse date '{date_str}', using as-is")
        return date_str

def yield_instances(dataset_path: str) -> Generator[LongMemInstance, None, None]:
    """Stream LongMemEval instances one at a time"""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    for item in dataset:
        # Convert sessions to structured Turn objects
        structured_sessions = []
        for session_idx, session in enumerate(item['haystack_sessions']):
            session_date = item['haystack_dates'][session_idx]
            turns = []
            for turn_data in session:
                turn = Turn(
                    role=turn_data['role'],
                    content=turn_data['content'],
                    iso_timestamp=parse_date_to_iso(session_date),
                    has_answer=turn_data.get('has_answer', False)
                )
                turns.append(turn)
            structured_sessions.append(turns)
        
        yield LongMemInstance(
            question_id=item['question_id'],
            question=item['question'],
            gold_answer=item['answer'],
            question_date=item['question_date'],
            question_type=item['question_type'],
            sessions=structured_sessions,
            haystack_dates=item['haystack_dates'],
            answer_session_ids=item.get('answer_session_ids', [])
        )

def load_instances(dataset_path: str, limit: Optional[int] = None) -> List[LongMemInstance]:
    """Load instances into a list (for small datasets)"""
    instances = []
    for i, instance in enumerate(yield_instances(dataset_path)):
        if limit and i >= limit:
            break
        instances.append(instance)
    return instances

def format_session_content(turns: List[Turn], session_date: str) -> str:
    """Format turns into a coherent session narrative"""
    content_parts = [f"Session Date: {session_date}\n"]
    
    for i, turn in enumerate(turns):
        if turn.content and turn.content.strip():
            content_parts.append(f"{turn.role.capitalize()} (Turn {i+1}): {turn.content}")
    
    return "\n\n".join(content_parts)

def get_session_stats(instance: LongMemInstance) -> dict:
    """Get statistics about a LongMemEval instance"""
    total_turns = sum(len(session) for session in instance.sessions)
    non_empty_turns = sum(1 for session in instance.sessions 
                         for turn in session if turn.content.strip())
    
    return {
        "question_id": instance.question_id,
        "question_type": instance.question_type,
        "total_sessions": len(instance.sessions),
        "total_turns": total_turns,
        "non_empty_turns": non_empty_turns,
        "avg_turns_per_session": total_turns / len(instance.sessions) if instance.sessions else 0
    } 