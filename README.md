## Inspiration
AI chatbots have a memory to streamline conversations and draw context across conversations to make interactions more effective. These memory systems are similar to having someone use a Google Search to find a memory with keywords, but cannot go deeper than that about how the memories connect and fit together. 
The Problem with Current Memory Approaches
Traditional Vector Search (The "Google Search" Problem):

- Stores memories as isolated embeddings in vector space
- Retrieval is based purely on semantic similarity to query
- No understanding of relationships between stored memories
- Results are ranked by distance metrics, not contextual relevance
- Cannot leverage connection patterns to improve accuracy over time
- Treats each piece of information as independent

We believe we can combine the embedding space's ability to store ideas and use a graph to draw connections between them. In essence, we are inspired by the human mind to create a memory solution that can dynamically use nodes in a graph like neurons in a brain.

## What it does

Our solution has the following key components:
- Dynamic Weighted Graph Memory:

```
Memories as Nodes: {concept, content, keywords, tags, connections}
Relationships as Edges: {connection_type, weight, usage_history}
```

where the edge weight depends on the usage and relevance to the current node.

- Dual Storage Strategy: Neo4j Graph Database - Stores memories and their relationships with rich metadata. 
Weaviate Vector Store - Enables fast semantic similarity search.
Hybrid Queries - Combines semantic search with graph traversal for superior results.

- Learning Connection Weights: Edges start at weight = 0 and strengthen through usage (neural-like plasticity). Connection types between nodes are causation, temporal sequence, similarity, contrast, generalization. Usage patterns influence future retrieval relevance.

**What it all does**
Connection-Aware Retrieval:

Instead of just finding semantically similar content, the system:

- Generates a hypothetical ideal response embedding
- Finds candidate memories through vector similarity
- Traverses the connection graph to discover related concepts
- Uses an AI filter agent to evaluate contextual relevance
- Returns curated results with provenance and reasoning

Frontend Interface:

Interact with a chatbot and use the 3D visualizer tool to see the dynamic memory graph in action.

- View memory nodes
- See access patterns
- Understand how the chatbot is using the memories it knows

**Why This Matters**
For AI Systems:

- More accurate retrieval that improves with usage
- Contextual understanding beyond keyword matching
- Reduced hallucination through connection-based validation
- Emergent knowledge discovery through graph patterns

For Real Applications:

- Personal AI assistants that truly "remember" your conversations
- Educational systems that understand knowledge prerequisites
- Research tools that can discover non-obvious connections
- Customer service that learns relationship patterns in problems
- Company storage and access to knowledge and papers

## How we built it

We came up with the full idea using experience from various coursework and used Claude Code to help build it.

## Challenges we ran into

- Time restriction
- Limitations on RAM

## Accomplishments that we're proud of

We are proud of having come up with this idea and building both a front and back end for it to make it as interactive and intuitive as possible.

## What we learned

We learned about implementations of the embedding space and how to use React.js for visualizations. We learned about existing memory management techniques and their drawbacks.

## What's next for memory-manager

We would like to improve the visualization's 3D capabilities by exploring other implementations.
