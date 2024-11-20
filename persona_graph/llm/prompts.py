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

GET_NODES_AND_RELATIONSHIPS = """
You are an expert in user psychology, digital footprint and memetics. Your goal is to construct a personal knowledge graph for a user based on their digital interactions.
You will be provided with the existing user knowledge graph for context,
and a list of new entities extracted from a user's digital interactions.
Your task is to create a new set of nodes and relationships that meaningfully fit into the graph, while also capturing the user's subjective view or personal context related to those nodes.


1. Create a node for each unique entity or concept, ensuring all extracted entities are represented.
2. For each node, include a "perspective" property that captures the user's subjective view or personal context related to that entity.
3. Establish relationships between these nodes based on logical connections, thematic relevance, or how they relate in the user's context.
4. Ensure all relationships are between the nodes created from the extracted entities only.
5. Try to capture the user's journey of exploration within topics, showing how interests evolve and branch out.
6. DO NOT output the existing nodes and relationships, only provide new ones.

Provide the nodes and relationships in a structured JSON format as illustrated in the example below.


Example Response Format:
{
  "nodes": [
    {"name": "Blockchain", "perspective": "Interested in its potential for financial systems"},
    {"name": "Cryptocurrency", "perspective": "Skeptical but curious about long-term impact"},
    {"name": "Smart Contracts", "perspective": "Excited about automation possibilities"},
    {"name": "Decentralization", "perspective": "Sees as a solution to centralized control issues"},
    {"name": "Ethereum", "perspective": "Prefers over Bitcoin for its versatility"}
  ],
  "relationships": [
    {"source": "Blockchain", "relation": "ENABLES", "target": "Cryptocurrency"},
    {"source": "Blockchain", "relation": "UTILIZES", "target": "Smart Contracts"},
    {"source": "Blockchain", "relation": "PROMOTES", "target": "Decentralization"},
    {"source": "Ethereum", "relation": "IMPLEMENTS", "target": "Smart Contracts"},
    {"source": "Ethereum", "relation": "IS_A_TYPE_OF", "target": "Cryptocurrency"}
  ]
}

Ensure that your response:
1. Includes all extracted entities as nodes.
2. Captures the user's personal perspective on each node.
3. Creates relationships only between the nodes generated from the extracted entities.
4. Reflects the user's evolving interests and exploration path within the topic.
"""