from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from .models import Message, Conversation
import tiktoken
from collections import deque


class ChatProcessor:
    def __init__(self, storage):
        self.storage = storage
        self.max_tokens = 4000
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.tokenizer.encode(text))
    
    async def add_message(self, user_id: str, conversation_id: str, message: Message) -> bool:
        """Add a message to an existing conversation"""
        try:
            return await self.storage.store_message(
                user_id,
                conversation_id,
                message
            )
        except Exception as e:
            print(f"Error processing message: {e}")
            return False

    async def truncate_conversation(self, conversation: Conversation, max_tokens: Optional[int] = None) -> Conversation:
        """Truncate conversation to fit within token limit while preserving recent context"""
        max_tokens = max_tokens or self.max_tokens
        messages = conversation.messages
        total_tokens = 0
        truncated_messages = []

        # Process messages in reverse (newest first)
        for msg in reversed(messages):
            tokens = self.count_tokens(msg.content)
            if total_tokens + tokens <= max_tokens:
                truncated_messages.append(msg)
                total_tokens += tokens
            else:
                break

        # Reverse back to chronological order
        truncated_messages.reverse()
        return Conversation(
            id=conversation.id,
            user_id=conversation.user_id,
            messages=truncated_messages,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            metadata=conversation.metadata
        )

    async def sliding_window_summary(self, conversation: Conversation, window_size: int = 5) -> List[Dict[str, Any]]:
        """Generate summaries for conversation windows"""
        messages = conversation.messages
        summaries = []
        
        if len(messages) <= window_size:
            return [{
                "start_time": messages[0].timestamp if messages else None,
                "end_time": messages[-1].timestamp if messages else None,
                "message_count": len(messages),
                "summary": "Conversation too short for windowing"
            }]

        window = deque(maxlen=window_size)
        current_summary = {
            "messages": [],
            "start_time": None,
            "end_time": None
        }

        for msg in messages:
            window.append(msg)
            
            if len(window) == window_size:
                current_summary = {
                    "start_time": window[0].timestamp,
                    "end_time": window[-1].timestamp,
                    "message_count": len(window),
                    "topics": await self._extract_topics(list(window)),
                    "sentiment": await self._analyze_sentiment(list(window))
                }
                summaries.append(current_summary)

        return summaries

    async def get_conversation_metrics(self, conversation: Conversation) -> Dict[str, Any]:
        """Calculate various metrics for the conversation"""
        messages = conversation.messages
        
        if not messages:
            return {"message_count": 0, "duration": 0, "avg_response_time": 0}

        metrics = {
            "message_count": len(messages),
            "user_messages": len([m for m in messages if m.role == "user"]),
            "assistant_messages": len([m for m in messages if m.role == "assistant"]),
            "duration": (messages[-1].timestamp - messages[0].timestamp).total_seconds(),
            "start_time": messages[0].timestamp,
            "end_time": messages[-1].timestamp,
        }

        # Calculate average response time
        response_times = []
        for i in range(1, len(messages)):
            if messages[i].role != messages[i-1].role:
                response_time = (messages[i].timestamp - messages[i-1].timestamp).total_seconds()
                response_times.append(response_time)

        metrics["avg_response_time"] = sum(response_times) / len(response_times) if response_times else 0
        
        return metrics

    async def _extract_topics(self, messages: List[Message]) -> List[str]:
        """Extract main topics from a window of messages"""
        # This is a placeholder - in production, you'd want to use an LLM or topic modeling
        return ["Topic extraction not implemented"]

    async def _analyze_sentiment(self, messages: List[Message]) -> str:
        """Analyze sentiment for a window of messages"""
        # This is a placeholder - in production, you'd want to use an LLM or sentiment analysis
        return "Sentiment analysis not implemented"

    async def filter_conversation_by_timerange(
        self, 
        conversation: Conversation,
        start_time: datetime,
        end_time: datetime
    ) -> Conversation:
        """Filter conversation messages within a specific time range"""
        filtered_messages = [
            msg for msg in conversation.messages
            if start_time <= msg.timestamp <= end_time
        ]
        
        return Conversation(
            id=conversation.id,
            user_id=conversation.user_id,
            messages=filtered_messages,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            metadata=conversation.metadata
        ) 