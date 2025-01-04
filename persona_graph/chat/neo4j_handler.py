from typing import List, Optional, Dict, Any
from datetime import datetime
from .interface import ChatStorageInterface
from .models import Message, Conversation
from persona_graph.core.neo4j_database import Neo4jConnectionManager
import uuid

class Neo4jChatStorage(ChatStorageInterface):
    def __init__(self):
        self.neo4j_manager = Neo4jConnectionManager()

    async def create_conversation(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new conversation node and connect it to user"""
        conversation_id = str(uuid.uuid4())
        query = """
        MATCH (u:User {id: $user_id})
        CREATE (c:Conversation {
            id: $conversation_id,
            user_id: $user_id,
            created_at: datetime($timestamp),
            updated_at: datetime($timestamp),
            metadata: $metadata
        })
        CREATE (u)-[:HAS_CONVERSATION]->(c)
        RETURN c.id
        """
        try:
            async with self.neo4j_manager.driver.session() as session:
                now = datetime.utcnow().isoformat()
                await session.run(
                    query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    timestamp=now,
                    metadata=metadata or {}
                )
                return conversation_id
        except Exception as e:
            print(f"Error creating conversation: {e}")
            return None

    async def get_recent_messages(self, user_id: str, conversation_id: str, limit: int = 10) -> List[Message]:
        """Get most recent messages from a conversation"""
        query = """
        MATCH (c:Conversation {id: $conversation_id, user_id: $user_id})-[:CONTAINS]->(m:Message)
        RETURN m
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """
        try:
            async with self.neo4j_manager.driver.session() as session:
                result = await session.run(
                    query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    limit=limit
                )
                messages = []
                async for record in result:
                    message_data = record["m"]
                    messages.append(Message(
                        role=message_data["role"],
                        content=message_data["content"],
                        timestamp=datetime.fromisoformat(message_data["timestamp"]),
                        metadata=message_data["metadata"]
                    ))
                messages.reverse()  # Restore chronological order
                return messages
        except Exception as e:
            print(f"Error retrieving recent messages: {e}")
            return []

    async def get_messages_by_timerange(
        self, 
        user_id: str, 
        conversation_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Message]:
        """Get messages within a specific time range"""
        query = """
        MATCH (c:Conversation {id: $conversation_id, user_id: $user_id})-[:CONTAINS]->(m:Message)
        WHERE datetime($start_time) <= m.timestamp <= datetime($end_time)
        RETURN m
        ORDER BY m.timestamp
        """
        try:
            async with self.neo4j_manager.driver.session() as session:
                result = await session.run(
                    query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat()
                )
                messages = []
                async for record in result:
                    message_data = record["m"]
                    messages.append(Message(
                        role=message_data["role"],
                        content=message_data["content"],
                        timestamp=datetime.fromisoformat(message_data["timestamp"]),
                        metadata=message_data["metadata"]
                    ))
                return messages
        except Exception as e:
            print(f"Error retrieving messages by timerange: {e}")
            return [] 