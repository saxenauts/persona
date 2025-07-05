"""
End-to-end tests for the new LLM client system integrated with the application.
"""

import pytest
import os
from unittest.mock import patch
from persona.llm.llm_graph import get_nodes, get_relationships, generate_response_with_context


@pytest.mark.skipif(
    not os.getenv("AZURE_API_KEY") or not os.getenv("AZURE_API_BASE"), 
    reason="Azure API credentials not available"
)
class TestLLME2EAzure:
    """End-to-end tests with Azure OpenAI"""
    
    @patch('persona.llm.client_factory.config')
    @pytest.mark.asyncio
    async def test_node_extraction_azure(self, mock_config):
        """Test node extraction using Azure OpenAI"""
        # Configure for Azure
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "azure/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.AZURE_API_KEY = os.getenv("AZURE_API_KEY")
        mock_config.MACHINE_LEARNING.AZURE_API_BASE = os.getenv("AZURE_API_BASE")
        mock_config.MACHINE_LEARNING.AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
        mock_config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
        mock_config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        # Test text for node extraction
        test_text = "I love playing guitar and reading science fiction books. I'm particularly interested in space exploration and dream of becoming an astronaut someday."
        graph_context = "Existing interests: music, technology"
        
        # Extract nodes
        nodes = await get_nodes(test_text, graph_context)
        
        # Verify we got some nodes
        assert len(nodes) > 0
        print(f"Extracted {len(nodes)} nodes:")
        for node in nodes:
            print(f"  - {node.name} ({node.type})")
            assert hasattr(node, 'name')
            assert hasattr(node, 'type')
            assert len(node.name) > 0
            assert len(node.type) > 0
    
    @patch('persona.llm.client_factory.config')
    @pytest.mark.asyncio
    async def test_relationship_generation_azure(self, mock_config):
        """Test relationship generation using Azure OpenAI"""
        # Configure for Azure
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "azure/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.AZURE_API_KEY = os.getenv("AZURE_API_KEY")
        mock_config.MACHINE_LEARNING.AZURE_API_BASE = os.getenv("AZURE_API_BASE")
        mock_config.MACHINE_LEARNING.AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
        mock_config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
        mock_config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        # Import the Node class from our new system
        from persona.llm.llm_graph import Node
        
        # Create test nodes
        nodes = [
            Node(name="Playing guitar", type="Hobby"),
            Node(name="Reading science fiction", type="Interest"),
            Node(name="Space exploration fascination", type="Interest"),
            Node(name="Astronaut dream", type="Goal")
        ]
        
        graph_context = "User has interests in music and technology"
        
        # Generate relationships
        relationships, id_mapping = await get_relationships(nodes, graph_context)
        
        # Verify we got some relationships
        print(f"Generated {len(relationships)} relationships:")
        for rel in relationships:
            print(f"  - {rel.source} --[{rel.relation}]--> {rel.target}")
            assert hasattr(rel, 'source')
            assert hasattr(rel, 'target')
            assert hasattr(rel, 'relation')
            assert len(rel.source) > 0
            assert len(rel.target) > 0
            assert len(rel.relation) > 0
        
        # Verify ID mapping was created
        assert len(id_mapping) == len(nodes)
        for temp_id, node_name in id_mapping.items():
            assert temp_id.startswith("Node")
            assert node_name in [node.name for node in nodes]
    
    @patch('persona.llm.client_factory.config')
    @pytest.mark.asyncio
    async def test_rag_response_azure(self, mock_config):
        """Test RAG response generation using Azure OpenAI"""
        # Configure for Azure
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "azure/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.AZURE_API_KEY = os.getenv("AZURE_API_KEY")
        mock_config.MACHINE_LEARNING.AZURE_API_BASE = os.getenv("AZURE_API_BASE")
        mock_config.MACHINE_LEARNING.AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
        mock_config.MACHINE_LEARNING.AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
        mock_config.MACHINE_LEARNING.AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        # Test query and context
        query = "What are my main interests?"
        context = """
        User Profile Context:
        - Enjoys playing guitar (Hobby)
        - Reads science fiction books (Interest)
        - Fascinated by space exploration (Interest)
        - Dreams of becoming an astronaut (Goal)
        - Has background in music and technology
        """
        
        # Generate response
        response = await generate_response_with_context(query, context)
        
        # Verify we got a meaningful response
        assert len(response) > 0
        assert isinstance(response, str)
        print(f"RAG Response: {response}")
        
        # Check that the response mentions some of the context
        response_lower = response.lower()
        assert any(keyword in response_lower for keyword in ['guitar', 'music', 'science', 'space', 'astronaut'])


class TestLLME2EMocked:
    """End-to-end tests with mocked LLM calls (for CI/CD)"""
    
    @patch('persona.llm.client_factory.config')
    @pytest.mark.asyncio
    async def test_node_extraction_json_parsing(self, mock_config):
        """Test that our JSON parsing works correctly with mocked responses"""
        from persona.llm.client_factory import get_chat_client, reset_clients
        from persona.llm.providers.base import ChatResponse
        from unittest.mock import AsyncMock
        
        # Configure for OpenAI with fake key
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "openai/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_API_KEY = "fake-key"
        mock_config.MACHINE_LEARNING.OPENAI_CHAT_MODEL = "gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        
        reset_clients()  # Reset to use new config
        
        # Mock the client to return a JSON response
        mock_client = AsyncMock()
        mock_client.chat.return_value = ChatResponse(
            content='{"nodes": [{"name": "Test Node", "type": "Test"}, {"name": "Another Node", "type": "Interest"}]}',
            model="mock-model"
        )
        
        with patch('persona.llm.client_factory.get_chat_client', return_value=mock_client):
            nodes = await get_nodes("Test text", "Test context")
            
            assert len(nodes) == 2
            assert nodes[0].name == "Test Node"
            assert nodes[0].type == "Test"
            assert nodes[1].name == "Another Node"
            assert nodes[1].type == "Interest"
    
    @patch('persona.llm.client_factory.config')
    @pytest.mark.asyncio
    async def test_relationship_generation_json_parsing(self, mock_config):
        """Test that relationship JSON parsing works correctly"""
        from persona.llm.client_factory import get_chat_client, reset_clients
        from persona.llm.providers.base import ChatResponse
        from persona.llm.llm_graph import Node
        from unittest.mock import AsyncMock
        
        # Configure for OpenAI with fake key
        mock_config.MACHINE_LEARNING.LLM_SERVICE = "openai/gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_API_KEY = "fake-key"
        mock_config.MACHINE_LEARNING.OPENAI_CHAT_MODEL = "gpt-4o-mini"
        mock_config.MACHINE_LEARNING.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        
        reset_clients()  # Reset to use new config
        
        # Mock the client to return a JSON response
        mock_client = AsyncMock()
        mock_client.chat.return_value = ChatResponse(
            content='{"relationships": [{"source_id": "Node1", "relation": "RELATES_TO", "target_id": "Node2"}]}',
            model="mock-model"
        )
        
        nodes = [
            Node(name="First Node", type="Test"),
            Node(name="Second Node", type="Test")
        ]
        
        with patch('persona.llm.client_factory.get_chat_client', return_value=mock_client):
            relationships, id_mapping = await get_relationships(nodes, "Test context")
            
            assert len(relationships) == 1
            assert relationships[0].source == "First Node"
            assert relationships[0].relation == "RELATES_TO"
            assert relationships[0].target == "Second Node"
            
            # Verify ID mapping
            assert id_mapping["Node1"] == "First Node"
            assert id_mapping["Node2"] == "Second Node" 