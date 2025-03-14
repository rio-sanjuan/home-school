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
## Graph Embedding for Simple Graphs

Here are some graph embedding algorithms for *simple* graphs (static, undirected, unsigned, and homogeneous). Algorithms are organized according to the information they attempt to preserve, including node co-occurence, structural role, node status, and community structure.
### Preserving Node Co-occurrence

One of the most popular ways to extract node co-occurrence in a graph is via performing random walks. Nodes are considered similar to each other if they tend to co-occur in these random walks. The [[Embedding#Mapping Function]] is optimized so that the learned node representations can reconstruct the "similarity" extracted from random walks.

### Mapping Function

A Direct way to define the mapping function $f(v_i)$ is using a lookup table. This means that we retrieve node $v_i$'s embedding $\textbf{u}_i$ given its index $i$. Specifically, the mapping function is implemented as $$ f(v_i) = \textbf{u}_i = \textbf{e}_i^\text{T}\textbf{W}, $$ where $\textbf{e}_i ∈ \{0,1\}^N$  with $N = |\mathcal{V}|$ is the one-hot encoding of the node $v_i$. $\textbf{W}^{N\times d}$ are the embedding parameters to be learned, where $d$ is the dimension of the embedding. The $i^{th}$ row of the matrix $\textbf{W}$ denotes the representations (or the embedding) of node $v_i$. Hence, the number of parameters in the mapping function is $N\times d$.

### Random Walk-Based Co-occurrence Extractor

Given a starting node $v^{(0)}$ in a graph $\mathcal{G}$, we randomly walk to one of its neighbors. We repeat this process from the node until $T$ nodes are visited. This random sequence of visited nodes is a random walk of length $T$ on the graph. We formally define a random walk as follows:

Let $\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}$ denote a connected graph. The probability of moving from node $v^{(t)}$ on the $t^{th}$ step of the random walk is defined by: $$ p(v^{t+1} \lvert v^{(t)}) = \begin{cases}
\frac{1}{d(v^{(t)})} & v^{(t+1)} ∈ \mathcal{N}(v^{(t)} \\
0 & \text{else}
\end{cases}$$where $d(v^{(t)})$ denotes the degree of node $v^{(t)}$ and $\mathcal{N}(v^{(t)})$ is the set of neighbors of $v^{(t)}$. In other words, the next node is randomly selected from the neighbors of the current node following a uniform distribution. We use a random walk generator to summarize the above process as follows: $$ \mathcal{W} = \text{RW}(\mathcal{G}, v^{(0)}, T),$$where $\mathcal{W} = (v^{(0)}, \ldots, v^{(T-1)})$ denotes the generated random walk where $T$ is the length of the random walk.

Random walks have been employed as a similarity measure in various tasks such as content recommendation and community detection. In [[Embedding#DeepWalk]], a set of short random walks is generated from a given graph, and node co-occurrence is extracted from these random walks. Next, we detail the process of generating the set of random walks and extracting co-occurrence from them.

To generate random walks that can capture the information of the entire graph, each node is considered as a starting node to generate $\gamma$ random walks. Therefore, there are $N\cdot\gamma$ random walks in total. These random walks can be treated as sentences in an "artificial language" where the set of nodes $\mathcal{V}$ is its vocabulary. The [[Skip-Gram Model]] in language modeling

## DeepWalk

DeepWalk is a graph representation learning algorithm that generates embeddings for nodes in a graph. It was introduced in the paper "DeepWalk: Online Learning of Social Representations" by Perozzi et al. in 2014. The algorithm is inspired by techniques from Natural Language Processing (NLP) and applies them to graphs, leveraging random walks and the [[Skip-Gram Model]] (popularized by [[Word2Vec]]).
## Node2Vec

## Graph2Vec
