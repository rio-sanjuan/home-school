## Embedding

Graph embedding aims to map each node in a given graph into a low-dimensional vector representation (or commonly known as node embedding) that typically preserves some key information of the node in the original graph. A node in a graph can be viewed from two domains: 
1. The original graph domain, where nodes are connected via edges (or the graph structure)
2. The embedding domain, where each node is represented as a continuous vector
From this two-domain perspective, graph embedding layers targets mapping each node from the graph domain to the embedding domain so that the information in the graph domain can be preserved in the embedding domain. Two questions arise:
1. What information needs to be preserved?
2. How do we preserve this information?
There are many answers to these questions. For the first question, embedding algorithms generally try to preserve any of the following:
1. A node's neighborhood information
2. A node's structural role
3. Community information
In answering the second question, many methods have been proposed, but they all share the same idea, which is to reconstruct the graph domain information to be preserved by using the node representations in the embedding domain. The intuition is that those good node representations should be able to reconstruct the information we desire to preserve. Therefore, the mapping can be learned by minimizing the reconstruction error. The general framework involves:
1. A mapping function, which maps the node from the graph domain to the embedding domain
2. An information extractor, which extracts the key information $I$ we want to preserve from the graph domain
3. A reconstructor to construct the extracted graph information $I'$ using the embeddings from the embedding domain
4. An objective based on the extracted information $I$ and the reconstructed information $I'$. Typically, we optimize the objective to learn all parameters involved in the mapping and/or reconstructor.
### Graph Embedding for Simple Graphs

Here are some graph embedding algorithms for *simple* graphs (static, undirected, unsigned, and homogeneous). Algorithms are organized according to the information they attempt to preserve, including node co-occurence, structural role, node status, and community structure.
#### Preserving Node Co-occurrence