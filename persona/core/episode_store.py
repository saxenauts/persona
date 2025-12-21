"""
Episode storage operations for the Neo4j backend.

Implements the EpisodeStore interface for storing and retrieving
narrative episodes with temporal linking.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID

from persona.core.interfaces import GraphDatabase
from persona.models.episode import Episode, EpisodeChainResponse
from server.logging_config import get_logger

logger = get_logger(__name__)


class EpisodeStore:
    """
    Manages Episode storage and temporal chain operations.
    
    Episodes are stored as nodes in the graph with PREVIOUS/NEXT edges
    forming a chronological chain (the "infinite necklace" of experience).
    """
    
    def __init__(self, graph_db: GraphDatabase):
        self.graph_db = graph_db
    
    async def create_episode(
        self, 
        episode: Episode,
        link_to_previous: bool = True
    ) -> Episode:
        """
        Create an episode and optionally link it to the previous episode in the chain.
        
        Args:
            episode: The Episode to create
            link_to_previous: If True, find and link to the most recent episode for this user
            
        Returns:
            The created Episode with any updated chain links
        """
        # Find the most recent episode for this user to link to
        if link_to_previous:
            previous = await self.get_most_recent_episode(episode.user_id)
            if previous:
                episode.previous_episode_id = previous.id
        
        # Create the episode node
        node_data = {
            "name": str(episode.id),
            "type": "Episode",
            "properties": {
                "id": str(episode.id),
                "title": episode.title,
                "content": episode.content,
                "timestamp": episode.timestamp.isoformat(),
                "day_id": episode.day_id,
                "session_id": episode.session_id,
                "source_type": episode.source_type,
                "source_ref": episode.source_ref,
                "access_count": episode.access_count,
                "last_accessed": episode.last_accessed.isoformat() if episode.last_accessed else None,
                "previous_episode_id": str(episode.previous_episode_id) if episode.previous_episode_id else None,
                "next_episode_id": str(episode.next_episode_id) if episode.next_episode_id else None,
            }
        }
        
        await self.graph_db.create_nodes([node_data], episode.user_id)
        
        # Create PREVIOUS/NEXT edges
        if episode.previous_episode_id:
            await self._create_temporal_edge(
                from_id=episode.id,
                to_id=episode.previous_episode_id,
                edge_type="PREVIOUS",
                user_id=episode.user_id
            )
            # Also update the previous episode's next_episode_id
            await self._update_next_link(
                episode_id=episode.previous_episode_id,
                next_id=episode.id,
                user_id=episode.user_id
            )
        
        logger.info(f"Created episode '{episode.title}' for user {episode.user_id}")
        return episode
    
    async def get_episode(self, episode_id: UUID, user_id: str) -> Optional[Episode]:
        """Retrieve a single episode by ID."""
        node_data = await self.graph_db.get_node(str(episode_id), user_id)
        
        if not node_data:
            return None
        
        return self._node_to_episode(node_data, user_id)
    
    async def get_most_recent_episode(self, user_id: str) -> Optional[Episode]:
        """Get the most recent episode for a user (tail of the chain)."""
        # This requires a custom query to find the episode with no next_episode_id
        # For now, we'll use a simpler approach via the interface
        # TODO: Add a more efficient query method to the interface
        
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        episodes = [
            self._node_to_episode(n, user_id) 
            for n in all_nodes 
            if n.get('type') == 'Episode'
        ]
        
        if not episodes:
            return None
        
        # Find the one with no next_episode_id (tail of chain)
        tail_episodes = [e for e in episodes if e.next_episode_id is None]
        
        if tail_episodes:
            # Return the most recent by timestamp
            return max(tail_episodes, key=lambda e: e.timestamp)
        
        # Fallback: just return the most recent by timestamp
        return max(episodes, key=lambda e: e.timestamp)
    
    async def get_episodes_by_day(self, day_id: str, user_id: str) -> List[Episode]:
        """Get all episodes for a specific day."""
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        
        episodes = []
        for node in all_nodes:
            if node.get('type') == 'Episode':
                props = node.get('properties', {})
                if props.get('day_id') == day_id:
                    episodes.append(self._node_to_episode(node, user_id))
        
        # Sort by timestamp
        episodes.sort(key=lambda e: e.timestamp)
        return episodes
    
    async def get_episode_chain(
        self, 
        user_id: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50
    ) -> EpisodeChainResponse:
        """
        Get a chain of episodes within a date range.
        
        Args:
            user_id: User to query
            start_date: Optional start day_id (YYYY-MM-DD)
            end_date: Optional end day_id (YYYY-MM-DD)
            limit: Maximum episodes to return
            
        Returns:
            EpisodeChainResponse with episodes in chronological order
        """
        all_nodes = await self.graph_db.get_all_nodes(user_id)
        
        episodes = []
        for node in all_nodes:
            if node.get('type') == 'Episode':
                episode = self._node_to_episode(node, user_id)
                
                # Apply date filters
                if start_date and episode.day_id < start_date:
                    continue
                if end_date and episode.day_id > end_date:
                    continue
                
                episodes.append(episode)
        
        # Sort chronologically
        episodes.sort(key=lambda e: e.timestamp)
        
        # Apply limit
        episodes = episodes[:limit]
        
        return EpisodeChainResponse(
            episodes=episodes,
            start_date=episodes[0].day_id if episodes else "",
            end_date=episodes[-1].day_id if episodes else "",
            total_count=len(episodes)
        )
    
    async def _create_temporal_edge(
        self, 
        from_id: UUID, 
        to_id: UUID, 
        edge_type: str,
        user_id: str
    ) -> None:
        """Create a PREVIOUS or NEXT edge between episodes."""
        relationship = {
            "source": str(from_id),
            "target": str(to_id),
            "relation": edge_type
        }
        await self.graph_db.create_relationships([relationship], user_id)
    
    async def _update_next_link(
        self, 
        episode_id: UUID, 
        next_id: UUID,
        user_id: str
    ) -> None:
        """Update an episode's next_episode_id (used when a new episode is added)."""
        # This would require an update operation - for now we'll handle via edges
        # The edge already captures the relationship
        await self._create_temporal_edge(
            from_id=episode_id,
            to_id=next_id,
            edge_type="NEXT",
            user_id=user_id
        )
    
    def _node_to_episode(self, node: Dict[str, Any], user_id: str) -> Episode:
        """Convert a graph node to an Episode model."""
        props = node.get('properties', {})
        
        return Episode(
            id=UUID(props.get('id', node.get('name'))),
            title=props.get('title', ''),
            content=props.get('content', ''),
            timestamp=datetime.fromisoformat(props['timestamp']) if props.get('timestamp') else datetime.utcnow(),
            day_id=props.get('day_id', ''),
            session_id=props.get('session_id'),
            source_type=props.get('source_type', 'conversation'),
            source_ref=props.get('source_ref'),
            access_count=props.get('access_count', 0),
            last_accessed=datetime.fromisoformat(props['last_accessed']) if props.get('last_accessed') else None,
            previous_episode_id=UUID(props['previous_episode_id']) if props.get('previous_episode_id') else None,
            next_episode_id=UUID(props['next_episode_id']) if props.get('next_episode_id') else None,
            user_id=user_id
        )
