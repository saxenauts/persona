#!/usr/bin/env python3
"""
Simple Persona API Example
=========================

Demonstrates:
1. Creating a test user
2. Ingesting a conversation in batches
"""

import requests
import os
from pathlib import Path

def split_conversation(text):
    """Split conversation into user-assistant pairs"""
    # Split on 'user:' and 'A:' markers
    parts = []
    current = []
    
    for line in text.split('\n'):
        if line.startswith('user:') or line.startswith('A:'):
            if current:
                parts.append('\n'.join(current))
            current = [line]
        elif line.strip():
            current.append(line)
            
    if current:
        parts.append('\n'.join(current))
        
    # Group into pairs (user + assistant)
    pairs = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            pairs.append(parts[i] + '\n' + parts[i+1])
            
    return pairs

def main():
    # API endpoints - when running inside docker, connect to the app service
    if os.environ.get("DOCKER_ENV") == "1":
        BASE_URL = "http://app:8000/api/v1"
    else:
        BASE_URL = "http://localhost:8000/api/v1"
    
    print(f"DOCKER_ENV: {os.environ.get('DOCKER_ENV')}")
    print(f"Using BASE_URL: {BASE_URL}")
    USER_ID = "test_user"
    
    # Get conversation file path relative to this script
    script_dir = Path(__file__).parent
    conversation_path = script_dir / "conversation.txt"
    
    # Copy conversation.txt if it doesn't exist
    if not conversation_path.exists():
        source_path = script_dir.parent / "tests" / "assets" / "conversation.txt"
        if source_path.exists():
            conversation_path.write_text(source_path.read_text())
            print(f"Copied conversation from {source_path}")
        else:
            print(f"Error: Could not find conversation at {source_path}")
            return

    # Read conversation
    conversation_text = conversation_path.read_text()
    conversation_pairs = split_conversation(conversation_text)
    
    print(f"Found {len(conversation_pairs)} conversation pairs")

    try:
        # 1. Create user
        print(f"\n1. Creating user '{USER_ID}'...")
        response = requests.post(f"{BASE_URL}/users/{USER_ID}")
        response.raise_for_status()
        print("User created successfully")

        # 2. Ingest conversation pairs
        print("\n2. Ingesting conversation in batches...")
        for i, conversation in enumerate(conversation_pairs, 1):
            print(f"\nIngesting batch {i}/{len(conversation_pairs)}...")
            response = requests.post(
                f"{BASE_URL}/users/{USER_ID}/ingest",
                json={
                    "title": f"Conversation Batch {i}",
                    "content": conversation,
                    "metadata": {}
                }
            )
            if response.status_code != 201:
                print(f"Error details: {response.text}")
            response.raise_for_status()
            print(f"Batch {i} ingested successfully")

        print("\nAll done! You can now:")
        print("1. Open Neo4j Browser at http://localhost:7474")
        print("2. Login with default credentials (neo4j/password)")
        print("3. Try this query to see your conversation graph:")
        print("   MATCH (n:NodeName {UserId: 'test_user'})-[r]-(m:NodeName {UserId: 'test_user'}) RETURN n,r,m")

    except requests.exceptions.RequestException as e:
        print(f"\nError: {str(e)}")
        if "Connection refused" in str(e):
            print("\nMake sure the Persona server is running:")
            print("1. Start all services: docker compose up -d")
            print("2. Wait a few seconds for services to be ready")
            print("3. Run this script again")

if __name__ == "__main__":
    main() 