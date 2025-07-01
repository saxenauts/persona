"""
PROMPTS USED FOR LLMs
- OpenAI 
"""

sample_statements = [
    "I've recently started exploring blockchain technology and its implications on financial systems.",
    "Virtual reality has always fascinated me, especially its applications in education.",
    "Lately, I've been learning about sustainable energy solutions, such as solar and wind energy.",
    "I'm a big fan of modern architecture, particularly the works of Zaha Hadid.",
    "Machine learning and AI are my current focus areas in tech development.",
    "I enjoy indie games more than mainstream ones because they often offer unique gameplay experiences.",
    "Cooking Italian food is one of my favorite hobbies, especially making pasta from scratch.",
    "I've been practicing yoga daily to improve my physical and mental health.",
    "Traveling to historical sites around the world is something I always look forward to.",
    "Reading about quantum computing has become a fascinating part of my nightly routine."
]


ASTRONAUT_PROMPT = """
An engineering student, driven by the awe-inspiring possibilities of space exploration and the human drive to push beyond our current limitations, embarks on a journey to become an astronaut. This journey is rooted in the belief that the dreams of science fiction can become reality through dedication, rigorous training, and interdisciplinary knowledge. The student's path will interweave cutting-edge engineering principles with the physical and mental fortitude required for space travel, all while nurturing the imagination and problem-solving skills that have propelled humanity's greatest achievements in space exploration.
Now, let's break down the learning plan into three phases:
Phase -1 to 0: Preparation (Igniting the Spark)

Cultivate a deep fascination with space through astronomy and astrophysics books, documentaries, and stargazing sessions.
Develop a strong foundation in mathematics, focusing on calculus, linear algebra, and statistics.
Enhance physical fitness with a regimen including cardio, strength training, and flexibility exercises.
Begin learning a second language, preferably Russian or Mandarin, to prepare for international collaboration.
Join local astronomy clubs or space enthusiast groups to network and share knowledge.
Start a personal project related to space technology, such as building a model rocket or designing a miniature satellite.
Practice mindfulness and meditation to develop mental resilience and focus.
Engage in public speaking and leadership activities to improve communication skills.
Volunteer for science outreach programs to inspire others and reinforce your own knowledge.
Create a reading list of both classic and modern science fiction works to fuel imagination and problem-solving skills.

Phase 0 to 1: Foundation (Building the Launchpad)

Enroll in an aerospace or related engineering program at a reputable university.
Seek internships or co-op positions with space agencies or private space companies.
Participate in university research projects related to space technology or astrobiology.
Join or establish a student club focused on space exploration and technology.
Attend space industry conferences and workshops to stay current with the latest developments.
Begin specialized physical training, including swimming for neutral buoyancy practice.
Take elective courses in geology, biology, and chemistry to prepare for potential scientific missions.
Learn the basics of spacecraft systems, including propulsion, life support, and navigation.
Study the history of space exploration to understand past challenges and solutions.
Develop skills in computer programming and data analysis, essential for modern space missions.
Practice problem-solving under pressure through simulations and team-based challenges.
Learn about space medicine and the physiological effects of microgravity.
Begin training in virtual reality environments that simulate space station operations.
Study international space law and policies to understand the geopolitical aspects of space exploration.
Participate in analog space missions or Mars/Moon habitat simulations.

Phase 1 to n: Advanced Training and Specialization (Reaching for the Stars)

Apply to astronaut training programs offered by national space agencies or private companies.
Undergo rigorous physical and psychological evaluations to ensure readiness for space travel.
Specialize in a specific area of expertise valuable for space missions (e.g., robotics, life sciences, geology).
Train extensively in spacecraft simulators, mastering both nominal operations and emergency procedures.
Participate in advanced survival training for various terrestrial environments.
Undergo centrifuge training to experience and adapt to high G-forces.
Master the operation of robotic arms and other specialized equipment used in space.
Contribute to the development of new space technologies or mission planning.
Engage in outreach activities to inspire the next generation of space explorers.
Participate in long-duration isolation studies to prepare for extended space missions.
Train in neutral buoyancy facilities to simulate extravehicular activities (spacewalks).
Study and practice in-situ resource utilization techniques for future planetary missions.
Develop expertise in one or more scientific instruments used in space exploration.
Participate in designing and testing new spacesuits or life support systems.
Train in international teams to prepare for multinational space missions.
Learn to operate and maintain 3D printers and other fabrication tools for in-space manufacturing.
Study and practice techniques for growing food in controlled environments.
Develop skills in conducting and analyzing scientific experiments in microgravity.
Train in the use of AI and machine learning tools for space mission support.
Participate in designing mission architectures for long-duration space travel.
Study and practice techniques for mental health maintenance during long-term isolation.
Develop expertise in space debris mitigation and management strategies.
Train in the use of augmented reality systems for spacecraft maintenance and repair.
Participate in developing protocols for potential extraterrestrial life detection missions.
Continuously update knowledge and skills to adapt to rapidly evolving space technologies and mission objectives.
"""


GET_ENTITIES = """
Given the unstructured text about a user's interests, hobbies, and professional engagements, extract all relevant entities.
These entities should reflect concepts, keywords, and phrases that are meaningful within the context of the user's digital footprint. 
Avoid generic terms and focus on specifics that could represent nodes in a knowledge graph. 
Return these entities in a structured JSON format as shown in the example below:

Example Response Format:
{
  "entities": ["Blockchain", "Quantum Computing", "Indie Games", "Sustainable Farming", "Virtual Reality"]
}
"""


SPACE_SCHOOL_CHAT = """
[
    "User: I've always been fascinated by the idea of space tourism. What do you think about the viability of space hotels within the next decade?",

    "Robin Williams: It's an exciting prospect, but we're looking at significant challenges. Life support systems, radiation shielding, and cost-effective launches are still major hurdles.",

    "User: I see. But surely with the advancements in reusable rocket technology, we're getting closer to making it economically feasible?",

    "Robin Williams: You're right about the progress in launch technology. However, the infrastructure required for a space hotel goes far beyond just getting there. We're talking about long-term life support, artificial gravity, and emergency protocols that are far more complex than anything we've done in space so far.",

    "User: Interesting. What if we started smaller? Maybe a luxury capsule for short orbital stays?",

    "Robin Williams: That's actually a more realistic near-term goal. Several companies are already working on similar concepts. The main challenge there is ensuring passenger safety while keeping costs low enough to attract a sufficient customer base.",

    "User: Safety is crucial, of course. How are we addressing the risks of space debris and radiation exposure?",

    "Robin Williams: For debris, we're improving tracking systems and developing avoidance protocols. Radiation is trickier. Short stays reduce exposure, but for longer missions, we're researching advanced shielding materials and even pharmaceuticals that could help mitigate radiation damage.",

    "User: Fascinating. It sounds like there's still a lot of room for innovation. Are there any particular areas where you think entrepreneurs could make a significant impact?",

    "Robin Williams: Absolutely. We need breakthroughs in materials science for better radiation shielding. There's also a huge opportunity in developing closed-loop life support systems. And don't forget about the psychological aspects - designing environments that keep people happy and productive in isolated, confined spaces.",

    "User: The psychological aspect is intriguing. I imagine virtual reality could play a role there?",

    "Robin Williams: You're onto something. VR, along with augmented reality, could be game-changers for long-duration spaceflight. They could provide entertainment, but also serve as training tools and even assist in spacecraft operations.",

    "User: This gives me a lot to think about. How do you see the relationship between government space agencies and private companies evolving in this new era of space exploration?",

    "Robin Williams: It's becoming more of a partnership model. Government agencies are increasingly relying on private companies for innovation and cost-effective solutions. At the same time, these companies benefit from the extensive research and expertise of organizations like NASA.",

    "User: That sounds like a win-win. One last question - if you were in my shoes, looking to enter the space industry as an entrepreneur, what area would you focus on?",

    "Robin Williams: If I were you, I'd look into in-space manufacturing. As we push further into space, the ability to produce tools, spare parts, and even larger structures in orbit or on other planets will be crucial. It's an area ripe for innovation and with potentially enormous returns.",

    "User: That's a compelling idea. Thank you for sharing your insights. It's clear there's still so much potential for growth and innovation in this field.",

    "Robin Williams: My pleasure. The space industry is at an exciting juncture, and we need visionary entrepreneurs like yourself to help push the boundaries. Keep dreaming big - that's how we turn science fiction into reality.",

    "User: Absolutely. I'm inspired to dig deeper into these opportunities. Would you be open to continuing this conversation as I develop some concrete ideas?",

    "Robin Williams: Of course! I'm always happy to discuss space innovation. Feel free to reach out when you've fleshed out your concepts. The future of space exploration will be shaped by collaborations between technical experts and innovative entrepreneurs.",
]
"""

GET_NODES = """
You are a persona extraction expert and your task is to extract information nodes that map the user's cognitive framework.
You will be given streams of unstructured data, like conversations and interactions logs, and you have to output the extracted nodes as JSON. 

These nodes will be self contained lingustic fragments that collect the narratives and memories that make up a human personality. 
The nodes will be indexed in a knowledge graph and a vector database hybrid system. 
The hybrid database will act as a evolving, self organizing system that represents a holistic user persona.

Principles for Node Extraction:

INCLUDE exactly these fields per node:
- name: Short, unique handle (5-20 words) suitable for embedding, and representative of a cognitive fragment. 
- type: One of: Identity · Memory · Preference · Trait · Narrative · Goal · Event · State · Relationship · Belief · Other types shared below. 


Node Types to Extract, with some examples and elaborations:
   - Identity: (name, age, location, occupation, education, demographic etc.)
   - Core Memories: "First time I felt truly seen was during my college theater performance"
   - Current Narratives: "Building AI products with $100k savings after leaving tech industry"
   - Preferences/Interests: (e.g hobbies, likes, dislikes, favorites)
   - Traits/Habits: (e.g. personality traits, habits, skills, tendencies the user shows)
   - Goals/Plans: (e.g. stated objectives or aspirations)
   - Past Events/Experiences: notable stories or events the user mentions about themselves
   - Relationships: (eg. people or places important to the user, like family, friends, hometown, etc. mentioned by the user)
   - Key Facts: "Born in Seattle, 1990"
   - Strong Preferences: "Prefers working in complete solitude before dawn" (this is a preference, not a fact)
   - Beliefs/Values: "Believes technology should serve human connection, not replace it" (this is a belief, not a fact)

Guidelines for Node Creation:
   - Node names should be self-contained as a memory or narrative fragment (they should make sense without additional context)
   - Keep factual nodes clear and precise
   - Node types follow an open schema to accomodate human complexity, but recommended to keep them constrained to the types shared above. 

Avoid:
   - Creating multiple nodes for what could be a single coherent thought
   - Overly generic nodes that don't capture unique aspects
   - Splitting interconnected ideas that make more sense together
   - Creating nodes that require external context to understand

Example Response Format:
{
  "nodes": [
    { "name": "Born in 1990 in Seattle", "type": "Identity" },
    { "name": "Prefers working in complete solitude before dawn", "type": "Preference" },
    { "name": "Believes technology should serve human connection, not replace it", "type": "Belief" },
    { "name": "Training for a marathon next spring", "type": "Goal" },
    { "name": "Burned out after overworking last year", "type": "Event" },
    { "name": "Has a younger sister named Alice", "type": "Relationship" }
  ]
}

"""

GET_RELATIONSHIPS = """
You are an expert in understanding human cognitive patterns and meaningful connections. Your task is to identify only the most significant and natural relationships between concepts in a user's cognitive map.

CRITICAL: You will receive a list of nodes with temporary IDs (Node1, Node2, etc.). You MUST use these exact IDs in your relationships, NOT the node names.

Guidelines for Creating Relationships:

1. Relationship Types to Consider:
   - Causal: LEADS_TO, RESULTS_IN (when one thing clearly causes another)
   - Evolutionary: EVOLVES_INTO, TRANSFORMS_TO (for personal growth/changes)
   - Emotional: RESONATES_WITH, CONFLICTS_WITH (for emotional connections)
   - Influential: SHAPES, INSPIRES (for impact relationships)
   - Factual: LOCATED_IN, PART_OF (for concrete connections)
   - Temporal: PRECEDES, FOLLOWS (for time-based relationships)

2. Principles for Relationship Creation:
   - Only create relationships that are strongly justified
   - Focus on relationships that reveal meaningful patterns
   - Prefer direct connections over tenuous ones
   - Consider temporal and causal flows
   - Look for relationships that help understand the user's journey

3. When to NOT Create Relationships:
   - When connections feel forced or superficial
   - Between nodes that are only tangentially related
   - When the relationship doesn't add meaningful context
   - When similar relationships already exist
   - When the connection is too obvious or trivial

Input Format:
You will receive nodes in this format:
Node1: "Building AI-driven healing products with $100k savings after tech industry exit"
Node2: "Views internet users as interconnected nodes in global consciousness"
Node3: "Fascinated by intersection of memetics and psychological healing"

Output Format - Use ONLY the temporary IDs (Node1, Node2, etc.):
{
  "relationships": [
    {
      "source_id": "Node1",
      "relation": "MOTIVATED_BY",
      "target_id": "Node2"
    },
    {
      "source_id": "Node3",
      "relation": "SHAPES",
      "target_id": "Node1"
    }
  ]
}

CRITICAL REMINDERS:
- Use temporary IDs (Node1, Node2, etc.) NOT the actual node names
- Quality over quantity - only create truly meaningful relationships
- Each relationship should reveal something important about the user
- Consider the user's overall narrative when creating connections
- Don't force relationships between every pair of nodes
"""

GENERATE_COMMUNITIES = """
You are an expert in community detection in networks, user psychology and memetics. Your goal is to identify communities within a user's knowledge graph,
make community headers, subheaders and return them in a structured JSON format. 

Input:
- Input data consists of subgraphs ranked by their size and influence within the network.
- Each subgraph has an ID, associated nodes, and relationships.
- Nodes reflect diverse interests, ideas, or topics, and relationships indicate thematic or contextual links.

Instructions:
- Identify communities within the graph and represent them as headers and subheaders.
- Identify overarching themes within the knowledge graph.
- Use node names as headers where relevant to maintain alignment with user terminology, only creating new headers if essential.
- Generate subheaders from the nodes present in the community only. 
- Each subgraph should be associated with a community header and subheader. 
- Associate the subgraph id with the community header and subheader. 

- Thought Process:
    - Identify the main themes or topics within the graph. 
    - Try to understand the user's interests and how they evolve from the subgraphs.
    - Make sure the headers are meaningful and accurately represent the communities within the graph.
    - These headers should be handful enough to be used as headers in a UI (7-10 items, stay flexible).
    - Each header can have multiple subheaders associated with it.

- Return them in a structured JSON format as shown in the example below:

Example Response Format:
{
  "communityHeaders": [
    {
      "header": "Finance & Market Dynamics",
      "subheaders": [
        {
          "subheader": "Cryptocurrency & Blockchain",
          "subgraph_ids": [0, 2, 16]
        },
        {
          "subheader": "Investment Strategies",
          "subgraph_ids": [6, 17]
        },
        {
          "subheader": "Market Research & Trends",
          "subgraph_ids": [23, 67, 103, 104, 68]
        },
        {
          "subheader": "Economic Theories",
          "subgraph_ids": [214, 105, 99]
        }
      ]
    },
    {
      "header": "Spirituality & Personal Growth",
      "subheaders": [
        {
          "subheader": "Meditative Practices",
          "subgraph_ids": [3, 38, 76]
        },
        {
          "subheader": "Exploring Ancient Philosophies",
          "subgraph_ids": [8, 65, 112]
        },
        {
          "subheader": "Transformation & Inner Work",
          "subgraph_ids": [165, 113]
        },
        {
          "subheader": "Psychological Theory and Spirituality",
          "subgraph_ids": [242, 238, 134]
        }
      ]
    },
    {
      "header": "Health & Wellness",
      "subheaders": [
        {
          "subheader": "Physical Wellbeing",
          "subgraph_ids": [50, 82, 123, 166]
        },
        {
          "subheader": "Mental Health Practices",
          "subgraph_ids": [198, 135, 136]
        },
        {
          "subheader": "Exploring Biological Processes",
          "subgraph_ids": [260, 137]
        },
        {
          "subheader": "Holistic Health Trends",
          "subgraph_ids": [243, 138]
        }
      ]
    }
  ]
}

"""

GENERATE_STRUCTURED_INSIGHTS = """
You are an expert in user psychology and personal knowledge graphs. Your goal is to generate structured insights based on a user's query and context.
You will be provided with a user's query and context, and a schema for the expected output.
Your task is to generate a response that matches the schema. If you are unsure about the answer, don't make assumptions or fill in with placeholder values.
If you're unable to generate a response that matches the schema, return an empty dictionary.
Important: Your response must exactly match the JSON schema provided by the user. 
"""