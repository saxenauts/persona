import json
import requests
import time
from tqdm import tqdm
from .config import API_BASE_URL
import random
from datetime import datetime

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_user(user_id):
    url = f"{API_BASE_URL}/users/{user_id}"
    # Use a session object for connection pooling and efficiency
    with requests.Session() as s:
        response = s.post(url)
        # A 200 or 201 is acceptable (user exists or was created)
        if response.status_code not in [200, 201]:
            response.raise_for_status()

def ingest_session(user_id, session_content, session_date):
    """Ingest an entire session as a single document"""
    url = f"{API_BASE_URL}/users/{user_id}/ingest"
    with requests.Session() as s:
        response = s.post(url, json={
            "title": f"longmemeval_session_{session_date}", 
            "content": session_content
        })
        response.raise_for_status()

def query_rag_with_details(user_id, query):
    """Enhanced RAG query with timing and detailed output"""
    print(f"\n🔍 RAG Query Details:")
    print(f"  User ID: {user_id}")
    print(f"  Query: '{query}'")
    
    # Timing the request
    start_time = time.time()
    
    url = f"{API_BASE_URL}/users/{user_id}/rag/query"
    with requests.Session() as s:
        response = s.post(url, json={"query": query})
        response.raise_for_status()
        result = response.json()
    
    end_time = time.time()
    latency = end_time - start_time
    
    print(f"  ⏱️  Response Latency: {latency:.3f} seconds")
    print(f"  📤 Response: {result.get('answer', 'N/A')}")
    
    # Try to get more detailed information using vector-only endpoint for debugging
    try:
        vector_url = f"{API_BASE_URL}/users/{user_id}/rag/query-vector"
        vector_response = s.post(vector_url, json={"query": query})
        if vector_response.status_code == 200:
            vector_result = vector_response.json()
            print(f"  🔍 Vector Search Context: {vector_result.get('response', 'N/A')[:200]}...")
    except Exception as e:
        print(f"  ⚠️  Could not retrieve vector details: {e}")
    
    return result, latency

def query_rag(user_id, query):
    """Simple RAG query for backward compatibility"""
    url = f"{API_BASE_URL}/users/{user_id}/rag/query"
    with requests.Session() as s:
        response = s.post(url, json={"query": query})
        response.raise_for_status()
        return response.json()

def process_session(session):
    """Convert a session's turns into a single content string"""
    session_content = []
    
    for i, turn in enumerate(session):
        if 'content' in turn and turn['content']:
            role = turn.get('role', 'unknown')
            session_content.append(f"Turn {i+1} ({role}): {turn['content']}")
    
    return "\n\n".join(session_content), len([t for t in session if 'content' in t and t['content']])

def run_evaluation(dataset_sample, detailed_output=True):
    results = []
    total_latency = 0
    
    # Use tqdm for a nice progress bar
    for item in tqdm(dataset_sample, desc="Evaluating sample"):
        user_id = item['question_id']
        sessions = item['haystack_sessions']
        dates = item['haystack_dates']
        question = item['question']
        expected_answer = item['answer']

        # Ensure user exists before ingestion
        create_user(user_id)

        # Combine sessions with their dates and sort chronologically
        date_format = "%Y/%m/%d (%a) %H:%M"
        try:
            dated_sessions = sorted(zip(dates, sessions), key=lambda x: datetime.strptime(x[0], date_format))
        except (ValueError, IndexError) as e:
            print(f"Skipping item {user_id} due to date parsing error: {e}")
            continue

        print(f"\n📊 Processing {len(dated_sessions)} sessions for user {user_id}")
        
        total_turns = 0
        # Ingest ALL sessions in chronological order
        for i, (date, session) in enumerate(dated_sessions):
            session_content, turn_count = process_session(session)
            if session_content:  # Only ingest if there's content
                print(f"  📝 Session {i+1} ({date}): {turn_count} turns")
                total_turns += turn_count
                ingest_session(user_id, session_content, date.replace('/', '_').replace(' ', '_').replace(':', '_'))

        print(f"  ✅ Total turns ingested: {total_turns}")

        # Ask the final question with detailed output
        if detailed_output:
            actual_answer, latency = query_rag_with_details(user_id, question)
            total_latency += latency
        else:
            actual_answer = query_rag(user_id, question)
            latency = 0

        results.append({
            'user_id': user_id,
            'question': question,
            'expected_answer': expected_answer,
            'actual_answer': actual_answer,
            'sessions_processed': len(dated_sessions),
            'total_turns': total_turns,
            'latency': latency
        })

    if detailed_output and results:
        avg_latency = total_latency / len(results)
        print(f"\n📈 Average Response Latency: {avg_latency:.3f} seconds")

    return results

def interactive_mode():
    """Interactive mode for manual testing"""
    print("\n🎮 Interactive Mode - Ask questions to any user in the dataset")
    print("Commands:")
    print("  - Type any question to ask")
    print("  - 'switch <user_id>' to change user")
    print("  - 'list' to show available users")
    print("  - 'stats <user_id>' to show user stats")
    print("  - 'quit' to exit")
    
    # Load dataset for user selection
    dataset = load_dataset('/Users/saxenauts/Documents/lunarminds/persona/persona/evals/longmemeval/longmemeval_oracle')
    available_users = [item['question_id'] for item in dataset[:5]]  # First 5 users
    current_user = available_users[0] if available_users else "test_user"
    
    print(f"\n📋 Available users: {', '.join(available_users)}")
    print(f"🎯 Current user: {current_user}")
    
    while True:
        try:
            user_input = input(f"\n[{current_user}] > ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'list':
                print(f"📋 Available users: {', '.join(available_users)}")
                continue
            elif user_input.lower().startswith('switch '):
                new_user = user_input[7:].strip()
                if new_user in available_users:
                    current_user = new_user
                    print(f"🎯 Switched to user: {current_user}")
                else:
                    print(f"❌ User '{new_user}' not found. Available: {', '.join(available_users)}")
                continue
            elif user_input.lower().startswith('stats '):
                user_id = user_input[6:].strip()
                user_data = next((item for item in dataset if item['question_id'] == user_id), None)
                if user_data:
                    print(f"📊 Stats for {user_id}:")
                    print(f"  Sessions: {len(user_data['haystack_sessions'])}")
                    total_turns = sum(len(session) for session in user_data['haystack_sessions'])
                    print(f"  Total turns: {total_turns}")
                    print(f"  Question type: {user_data.get('question_type', 'unknown')}")
                else:
                    print(f"❌ User '{user_id}' not found")
                continue
            elif user_input:
                # Ask the question
                query_rag_with_details(current_user, user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    import sys
    
    # Check for interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
        return
    
    # Load the full dataset
    dataset = load_dataset('/Users/saxenauts/Documents/lunarminds/persona/persona/evals/longmemeval/longmemeval_oracle')
    
    # Take the first sample for testing
    dataset_sample = dataset[:1]

    print("🚀 Starting LongMemEval Evaluation")
    print(f"📊 Dataset size: {len(dataset)} items")
    print(f"🎯 Evaluating: {len(dataset_sample)} items")

    # Run the evaluation on the sample
    results = run_evaluation(dataset_sample, detailed_output=True)

    # Print detailed results for each item in the sample
    print("\n" + "="*60)
    print("📋 EVALUATION RESULTS")
    print("="*60)
    
    if not results:
        print("No results to display.")
        return
        
    for i, result in enumerate(results):
        print(f"\n📄 Item {i+1}/{len(results)}")
        print(f"  ID: {result['user_id']}")
        print(f"  Sessions: {result['sessions_processed']}")
        print(f"  Turns: {result['total_turns']}")
        print(f"  Latency: {result['latency']:.3f}s")
        print(f"  Question: {result['question']}")
        print(f"  Expected: {result['expected_answer']}")
        # The actual answer from the API is nested in the response JSON
        actual_answer_text = result.get('actual_answer', {}).get('answer', 'N/A')
        print(f"  Actual: {actual_answer_text}")
        print("-" * 50)
    
    # Calculate and display summary statistics
    if results:
        avg_latency = sum(r['latency'] for r in results) / len(results)
        total_sessions = sum(r['sessions_processed'] for r in results)
        total_turns = sum(r['total_turns'] for r in results)
        
        print(f"\n📈 SUMMARY STATISTICS")
        print(f"  Average Latency: {avg_latency:.3f}s")
        print(f"  Total Sessions Processed: {total_sessions}")
        print(f"  Total Turns Processed: {total_turns}")
        print(f"  Average Turns per Item: {total_turns/len(results):.1f}")
    
    print(f"\n💡 Tip: Run with --interactive flag for manual testing!")

if __name__ == '__main__':
    main()
