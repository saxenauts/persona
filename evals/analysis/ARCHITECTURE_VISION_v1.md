# Persona Architecture Vision v1
> User's verbatim notes from Dec 26, 2025 session

---

## 1. Goals → Notes/Lists Rename

Need to make a fact like thing in graph, for all kinds of lists, and rename goal as list (or a better phrase?) It will contain different types. Goals, Projects, Tasks, Subtasks, Facts, Favorites, Budget, Logs, Contacts, etc. This is a big change that will affect memory components everywhere in terms of naming and other implications for data models, and for documentation and mintlify, and other places in adapters. Basically throughout. But the earlier we do it better. Lets call it Notes or Lists because this is where facts go, structured or unstructured. You know instead of journalling, I just make a two para note on what I think the future of consumer AI will be. It will get processed in the episode yes, but more importantly it is a list, or a note. Research reminders, ideas, etc. We must make a commit just for this particular change throughout (exhaustively this repo and mintlify one and all docs). And we do rest of the work only after this commit.

**Action**: Exhaustive rename commit across persona repo + mintlify docs before any other work.

---

## 2. Smarter Ingestion with Links

We currently do only one ingestion which outputs memory types. There is no link there. The only link we make are basic manual, back and forth, and episodes and psyche of the day. This is not enough, its level 1. The ingestion needs to be smarter as a process. It has to be sync/async hybrid I guess. We need to develop smart links in that first go. In the first ingestion call, we can give it more context. It already has conversation thread, and should have more stuff from past two days. So it also develops links. Or we can keep this part smart in the first ingestion go, as in LLM generates memory and we develop smart links without calling LLM. And then another call goes which builds the right context to review the first set of links but then make better calls to retrieve better memories to mutate better links. In a way thats useful. Whats useful needs to be thought through. We can learn from benchmarks, on how to give LLM instructions to build new links. We can take inspiration from neuroscience. Or we can try to make a prompt that gets the meta idea but can keep it emergent and evolving and self correcting itself. This will be true memory personalisation. Because forgetting needs to be intelligent too, and we will have updates dream like events for better consolidation too. Those are mechanisms we will add too but more on that later. We need to think through this architecturally and do some research on this. Iteratively.

**Key Ideas**:
- Multi-pass ingestion (sync/async hybrid)
- First pass: LLM generates memories
- Context includes: conversation thread + past 2 days
- Smart links without extra LLM call OR second pass for link refinement
- Emergent, self-correcting, evolving memory
- Intelligent forgetting

---

## 3. Intelligent Retrieval

We need to add more intelligent retrieval. I dont know how or what is going wrong.

**Action**: Investigate current retrieval failures, understand patterns.

---

## 4. Consolidation & Clustering

We need consolidation. We need to cluster. We need to make groups that are relevant to major themes in life, or project. Truth is there are not more than 2-3 themes running in life for most people. But we have to have clusters that can add to these groups/clusters in a more fluid way as well for purposes of more learning and checkpointing progress. We must read other research and figure out how this can be made smarter with LLM calls. This consolidation can run once after the previous steps of ingestion is done. But this consolidation will be designed in a way that we can give it flexible control or triggers, so different product designs can do what they want with it. We would do it every night. Opposite to the time zone of the User.

**Key Ideas**:
- 2-3 major life themes at any time
- Fluid cluster membership
- Runs during "dream phase" (user's night)
- Flexible triggers for different products
- LLM-powered clustering

---

## 5. User Profile One-Pager

Beyond fluid facts memories, we need like a one pager on the user with a maintained current theme and context in some paragraphs. And this should contain basic information, who what where how, etc. This is static, we need to figure out what agent or when does this gets rewired. During consolidation maybe? And it can also contain an index for smarter queries in the graph. So that the agent can make really intelligent queries and we retrieve nodes beyond just doing semantic similarity. We must research the best ways to do this, because I have learned that silly tavern and other companion app roleplays, store these memories as a JSON-like thing but in markdown, and retrieve it automatically with certain keywords for each such memory (nodes, but we want to call memory and links, instead of nodes and relationships).

**Key Ideas**:
- Static user summary (who/what/where/how)
- Current theme/context paragraphs
- Acts as query index for smarter retrieval
- Research SillyTavern's approach: JSON-in-markdown with keyword triggers
- Terminology: "memories and links" not "nodes and relationships"

---

## 6. Time & Chronology

Time and Chronology is a very important topic in memory, how it is implemented. We humans experience time very differently and subjectively, like time for me ran much faster in last two years than it did in two years before that. But with LLMs being the new interface that can semantically and conceptually become an extension of human lives and connect with them better and take load off of them that would be awesome, but computers today understand facts only, and most systems retrieve and log with date and time and so we should work with that, and have basic versioning in our memory regardless (TODO for later), but we should also have a fluid way so user can refer to events that happened (last week, yesterday, day before, maybe next year, in 2004, oh the year after my wedding, etc. things like that) LLM should be able to take natural language and make queries to smartly. Or we can ingest smartly. How and when this happens, I dont know, but in agentic retrieval step, it can happen for as many hops as needed. It can happen during ingestion too. Or consolidation.

**Key Ideas**:
- Natural language time references ("year after my wedding")
- Basic versioning (TODO)
- Agentic retrieval can resolve temporal references
- Can happen at ingestion, retrieval, or consolidation

---

## 7. Causal Chains

We need our links between memories to mean something. They should be fluid, but should be done in a way that memory can be subjective and emergent and change with time, we will have intelligence in the link, to add causal understanding to it as well, and follow the idea of fire together wire together and take what we can in latest computational neuroscience and see what can inspire us to develop these causal reasoning chains, which can also act as index for smarter and faster retrievals later.

**Key Ideas**:
- Meaningful, typed links
- Fluid and emergent over time
- Causal understanding in links
- "Fire together, wire together" principle
- Links as retrieval index

---

## Meta-Principles

> Most of it boils down to a great prompt design, and context engineering design. They go hand in hand and if we can learn anything from the eval design suggestions by hamel, it would be that we need to iterate on these changes, with smart groupings of feature adds, but to eventually be able to do iterations fast enough with evals to record analyze and log each update and design decision. SO we need to keep it shorter retrieval as well. Not just failures or where we lagged but other question types as well, for a fair and balanced analysis.

**Approach**:
- Fast iteration with evals
- Log every update and design decision
- Balanced analysis across all question types
- Prompt design + context engineering go together

---

## Research Targets

1. Honcho's approach to memory
2. Graphiti's edge-based model
3. SillyTavern's character memory system
4. Computational neuroscience: consolidation, Hebbian learning
5. BEAM benchmark design
6. Other memory systems (Mem0, Letta/MemGPT)

---

*Document created: Dec 26, 2025*
*Branch: arch/v1-release*
