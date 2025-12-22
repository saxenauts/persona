"""
Pydantic schemas for deep logging

Defines structured logging schemas for capturing all aspects of
evaluation runs, from ingestion through generation and evaluation.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SeedNode(BaseModel):
    """A seed node from vector search"""
    node_id: str
    score: float
    node_type: str  # 'episode', 'psyche', 'goal', etc.
    content: Optional[str] = None


class VectorSearchLog(BaseModel):
    """Logs from vector search phase"""
    top_k: int
    seeds: List[SeedNode]
    duration_ms: float


class GraphTraversalLog(BaseModel):
    """Logs from graph traversal phase"""
    max_hops: int
    nodes_visited: int
    relationships_traversed: int
    final_ranked_nodes: List[str]
    duration_ms: float


class RetrievalLog(BaseModel):
    """Complete retrieval logs"""
    query: str
    duration_ms: float
    vector_search: VectorSearchLog
    graph_traversal: GraphTraversalLog
    context_size_tokens: int
    retrieved_context: Optional[str] = None


class MemoryCreationStats(BaseModel):
    """Statistics about created memories"""
    episodes: int = 0
    psyche: int = 0
    goals: int = 0
    events: int = 0


class IngestionLog(BaseModel):
    """Logs from conversation ingestion phase"""
    duration_ms: float
    sessions_count: int
    memories_created: MemoryCreationStats
    nodes_created: int
    relationships_created: int
    embeddings_generated: int
    errors: List[str] = Field(default_factory=list)


class GenerationLog(BaseModel):
    """Logs from answer generation phase"""
    duration_ms: float
    model: str
    temperature: float
    prompt_tokens: int
    completion_tokens: int
    answer: str


class EvaluationLog(BaseModel):
    """Logs from evaluation/scoring phase"""
    gold_answer: str
    correct: bool
    judge_response: Optional[str] = None
    judge_model: Optional[str] = None
    score_type: str  # 'binary' or 'exact_match'


class QuestionLog(BaseModel):
    """Complete log for a single question evaluation"""
    question_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_id: str
    benchmark: str  # 'longmemeval' or 'personamem'
    question_type: str
    question: str

    ingestion: IngestionLog
    retrieval: RetrievalLog
    generation: GenerationLog
    evaluation: EvaluationLog

    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "question_id": "gpt4_abc123",
                "timestamp": "2024-12-21T14:30:52Z",
                "user_id": "Persona_q15_a8f3",
                "benchmark": "longmemeval",
                "question_type": "multi-session",
                "question": "How many times did I visit the gym?",
                "ingestion": {
                    "duration_ms": 15420,
                    "sessions_count": 47,
                    "memories_created": {
                        "episodes": 52,
                        "psyche": 18,
                        "goals": 7
                    },
                    "nodes_created": 77,
                    "relationships_created": 134,
                    "embeddings_generated": 77,
                    "errors": []
                },
                "retrieval": {
                    "query": "How many times did I visit the gym?",
                    "duration_ms": 1847,
                    "vector_search": {
                        "top_k": 5,
                        "seeds": [
                            {
                                "node_id": "episode_42",
                                "score": 0.94,
                                "node_type": "episode"
                            }
                        ]
                    },
                    "graph_traversal": {
                        "max_hops": 2,
                        "nodes_visited": 23,
                        "relationships_traversed": 45,
                        "final_ranked_nodes": ["episode_42", "episode_38"],
                        "duration_ms": 150
                    },
                    "context_size_tokens": 3452
                },
                "generation": {
                    "duration_ms": 2310,
                    "model": "gpt-4.1-mini",
                    "temperature": 0.7,
                    "prompt_tokens": 3580,
                    "completion_tokens": 87,
                    "answer": "You visited the gym 3 times."
                },
                "evaluation": {
                    "gold_answer": "3",
                    "correct": True,
                    "judge_response": "yes",
                    "judge_model": "gpt-4o",
                    "score_type": "binary"
                }
            }
        }


class RunMetadata(BaseModel):
    """Metadata for an entire evaluation run"""
    run_id: str
    benchmark: str
    started_at: str
    completed_at: Optional[str] = None
    total_questions: int
    questions_completed: int = 0
    questions_failed: int = 0
    overall_accuracy: Optional[float] = None
    config: Dict[str, Any] = Field(default_factory=dict)
