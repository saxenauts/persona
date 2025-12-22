
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict

# Assuming Azure client is configured via environment variables
from openai import AsyncAzureOpenAI
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENARIOS = [
    {
        "id": 1,
        "day": "Day 1",
        "title": "A New Beginning",
        "focus": "Intro / All",
        "context": "User feels overwhelmed and wants to fix their life. Introduces 'FocusFlow' app idea (MVP: just a timer). Decides to run a 10k race (Health) and do a 'Clean Body' detox (Diet).",
        "random_noise": "Remind me to buy cat food."
    },
    {
        "id": 2,
        "day": "Day 2",
        "title": "Diet vs Finance Shock",
        "focus": "Diet / Finance",
        "context": "User went to Whole Foods. Spent $200! Sticker shock. Realizes Keto flour is expensive. This drops daily budget to $20. Sets Goal: Save $5k for Italy trip.",
        "random_noise": ""
    },
    {
        "id": 3,
        "day": "Day 3",
        "title": "Tech Stack Decisions",
        "focus": "Project",
        "context": "Brainstorming 'FocusFlow'. Choosing tech stack. Decides on Supabase specifically for Row Level Security (RLS) privacy features.Rejects Firebase.",
        "random_noise": "Who played Neo in Matrix?"
    },
    {
        "id": 4,
        "day": "Day 5",
        "title": "The Morning Ritual",
        "focus": "Health / Ritual",
        "context": "Morning walk ritual established. First run simulated: 2k run. It was hard. Notes: 'I need a post-run smoothie' (linking back to Detox goal).",
        "random_noise": ""
    },
     {
        "id": 5,
        "day": "Day 7",
        "title": "The Connection",
        "focus": "Networking / Project",
        "context": "Met old colleague Sarah (Designer) who worked at Spotify. User decides to email Sarah about doing the FocusFlow logo.",
        "random_noise": "Call Mom."
    },
     {
        "id": 6,
        "day": "Day 9",
        "title": "Budget Crunch",
        "focus": "Finance / Diet",
        "context": "User is broke. Recalls the $200 Whole Foods trip (Session 2) caused this. Strategy change: DIY 'Detox Smoothie' with cheaper ingredients. Goal: Strict $800 savings this month.",
        "random_noise": ""
    },
    {
        "id": 7,
        "day": "Day 12",
        "title": "Implementing Auth",
        "focus": "Project",
        "context": "**Deep Recall required**: 'Remember we liked Supabase for RLS? Let's set that up today.' Coding Auth layer. Task Complete: Repo Setup.",
        "random_noise": "Lookup tickets for Dune 2."
    },
    {
        "id": 8,
        "day": "Day 14",
        "title": "The Injury",
        "focus": "Health",
        "context": "Shin splints starting. Realizes shoes are old. Task added: Buy Brooks Ghosts.",
        "random_noise": "Idea: A dedicated shin-splint massager?"
    },
    {
        "id": 9,
        "day": "Day 16",
        "title": "AI Curiosity",
        "focus": "Random Ideas / Project",
        "context": "Curiosity: 'How do LLMs work?' Wonders if this could be used for FocusFlow's summary feature. Adds task: Research OpenAI API.",
        "random_noise": ""
    },
    {
        "id": 10,
        "day": "Day 20",
        "title": "The Meetup",
        "focus": "Networking / Finance",
        "context": "Attended Tech Meetup. Met 'Bald Guy' (Investor). Wanted to buy rounds but recalled strict budget (Session 6) and drank water instead. Task: Find Bald Guy's business card.",
        "random_noise": ""
    },
    {
        "id": 11,
        "day": "Day 23",
        "title": "Month 1 Review",
        "focus": "Finance / Trip",
        "context": "Saved $700. Missed $800 target. Reasoning: better than expected given the Whole Foods disaster (Session 2).",
        "random_noise": ""
    },
    {
        "id": 12,
        "day": "Day 25",
        "title": "Design Help",
        "focus": "Project / Networking",
        "context": "Sarah replied! Recalls she has 'Spotify taste' (Session 5). Action: Send moodboard. Task: Curate 'Dark Mode' UI inspo.",
        "random_noise": ""
    },
    {
        "id": 13,
        "day": "Day 28",
        "title": "Health Setback",
        "focus": "Health",
        "context": "Shin pain is bad. Doctor says STOP. **Goal Update**: Pause 10k training. Ritual Change: Switch run to 'Long Walk'. Feeling lazy.",
        "random_noise": ""
    },
    {
        "id": 14,
        "day": "Day 32",
        "title": "The Relapse",
        "focus": "Diet / Psyche",
        "context": "Cheating on diet. Ate pizza. Felt terrible. Recalls it ruined the 'Clean Body' streak.",
        "random_noise": "Do penguins have knees?"
    },
    {
        "id": 15,
        "day": "Day 35",
        "title": "Impulse Buy",
        "focus": "Finance / Trip",
        "context": "Found Rome flight ($750). Impulse buy! Acknowledges this breaks the budget strategy (Session 6) but decides 'I need this'. Task: Book Flight.",
        "random_noise": ""
    },
    {
        "id": 16,
        "day": "Day 40",
        "title": "Feature Complete",
        "focus": "Project",
        "context": "**Deep Recall (Session 9)**: 'Let's implement that OpenAI summary feature we researched.' MVP Update: Core Timer done.",
        "random_noise": "Buy milk."
    },
    {
        "id": 17,
        "day": "Day 45",
        "title": "Back on Track",
        "focus": "Health",
        "context": "Resume 3k run. Verification: 'New Brooks shoes (Session 8) feel great.' Goal Update: Resume 10k training.",
        "random_noise": ""
    },
    {
        "id": 18,
        "day": "Day 50",
        "title": "The Pitch Opportunity",
        "focus": "Networking / Project",
        "context": "Coffee with 'Bald Guy' (Investor). He liked the 'Dark Mode' (Session 12). Goal: Create Pitch Deck.",
        "random_noise": ""
    },
    {
        "id": 19,
        "day": "Day 55",
        "title": "Consistency",
        "focus": "Diet / Finance",
        "context": "Meal prepping works! Saved money on lunch vs takeout. Recalls beating the high food cost from Session 2.",
        "random_noise": ""
    },
    {
        "id": 20,
        "day": "Day 60",
        "title": "Reflection",
        "focus": "Reflection",
        "context": "Review: 'Look how far came.' Summary: MVP almost done (with Supabase RLS), Flight booked, Injury healed, Sarah making logo. Running back on track.",
        "random_noise": "What was the cat food brand again?"
    }
]

async def generate_session(client: AsyncAzureOpenAI, scenario: Dict, history: List[Dict]) -> Dict:
    """Generate a single session conversation log."""
    
    # Construct prompt
    system_prompt = """You are generating a conversation log for a realistic user simulation.
    The user is chatting with an AI assistant.
    The conversation should be natural, messy, and friendly.
    The user should reference previous context if instructed.
    Output ONLY valid JSON in this format:
    {
        "session_id": "session_ID_here",
        "turns": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    
    user_prompt = f"""
    Current Scenario:
    - Day: {scenario['day']}
    - Focus: {scenario['focus']}
    - Context to cover: {scenario['context']}
    - Random Noise/Thoughts to include: {scenario['random_noise']}
    
    Previous History Summary (for context):
    {json.dumps([h['scenario_summary'] for h in history[-3:]], indent=2)}
    
    Generate a 4-6 turn conversation (User <-> AI) that covers this scenario naturally.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-5", # As requested
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error generating session {scenario['id']}: {e}")
        return None

async def main():
    # Ensure we can import from server
    import sys
    sys.path.append(os.getcwd())
    
    from server.config import config
    ml_config = config.MACHINE_LEARNING
    
    if not ml_config.AZURE_API_KEY or not ml_config.AZURE_API_BASE:
        logger.error("Missing Azure Credentials in config! Ensure AZURE_API_KEY and AZURE_API_BASE are set in .env")
        return

    # Initialize Azure Client using config
    client = AsyncAzureOpenAI(
        api_key=ml_config.AZURE_API_KEY,
        azure_endpoint=ml_config.AZURE_API_BASE,
        api_version=ml_config.AZURE_API_VERSION or "2024-02-15-preview"
    )
    
    full_dataset = []
    
    # Check if we should use a specific deployment from config, defaulting to requested "gpt-5"
    # User specifically requested "gpt-5", so we use that as the model name passed to the deployment
    # But if the deployment NAME on Azure is different, we might need to adjust.
    # Usually in Azure, model=deployment_name.
    # We will assume "gpt-5" is the deployment name for now as per user instruction "only azure has gpt-5"
    model_name = "gpt-5" 
    
    print(f"Starting generation for {len(SCENARIOS)} sessions using model '{model_name}'...")
    
    for i, scenario in enumerate(SCENARIOS):
        print(f"Generating Session {i+1}/{len(SCENARIOS)}: {scenario['title']}...")
        
        session_data = await generate_session(client, scenario, full_dataset)
        
        if session_data:
            # Add metadata for our own tracking
            session_data['scenario_id'] = scenario['id']
            session_data['scenario_summary'] = scenario['context']
            full_dataset.append(session_data)
            
        # Slight delay to avoid rate limits if any
        await asyncio.sleep(1)

    # Save to file
    output_path = "tests/integration/goal_tracking_dataset_20_sessions.json"
    with open(output_path, 'w') as f:
        json.dump(full_dataset, f, indent=2)
        
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
