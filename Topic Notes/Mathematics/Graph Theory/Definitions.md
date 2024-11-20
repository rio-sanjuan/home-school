Graphs, which describe pairwise relations between entities, are essential representations for real-world data from many different domains, including social science, linguistics, chemistry, biology, and physics. 
## Degree
In a graph $\mathcal{G} = \{\mathcal{V,E}\}$, the degree of a node $v_i\in\mathcal{V}$ is the number of nodes that are adjacent to $v_i$: $$ d(v_i) = \sum_{v_j\in\mathcal{V}}\mathbb{1}_\mathcal{E}(\{v_i, v_j\})$$ where $\mathbb{1}_\mathcal{E}(\cdot)$ is an indicator function: $$ \mathbb{1}_\mathcal{E}(\{v_i, v_j\}) = \begin{cases} 1 & \text{if } (v_i,v_j)\in\mathcal{E} \\ 0 & \text{if } (v_i,v_j)\notin\mathcal{E}\end{cases} $$ The degree of a node $v_i$ in $\mathcal{G}$ can also be calculated from its adjacency matrix.
## Neighbors
For a node $v_i$ in a graph $\mathcal{G} = \{\mathcal{V,E}\}$, the set of its neighbors $\mathcal{N}(v_i)$ consists of all nodes that are adjacent to $v_i$. Note that for a node $v_i$, the number of nodes in $\mathcal{N}(v_i)$ equals its degree: i.e., $d(v_i) = |\mathcal{N}(v_i)|$.
## Walk
A walk on a graph is an alternating sequence of nodes and edges, starting with a node and ending with a node where each edge is incident with the nodes immediately preceding and following it.

A walk starting at node $u$ and ending at node $v$ is called a $u-v$ walk. The length of a walk is the number of edges in this walk. Note that $u-v$ walks are not unique because there exist various $u-v$ walks with different lengths.

For a graph $\mathcal{G} = \{\mathcal{E,V}\}$ with the adjacency matrix $\textbf{A}$, we use $\textbf{A}^n$ to denote the $n^{th}$ power of the adjacency matrix. The $(i,j)^{th}$ element of the matrix $\textbf{A}^n$ equals to the number of $v_i-v_j$ walks of length $n$.
### Trail
A trail is a walk whose edges are distinct.
### Path
A path is a walk whose nodes are distinct.
## Subgraph
A subgraph $\mathcal{G}' = \{\mathcal{V',E'}\}$ of a given graph $\mathcal{G} = \{\mathcal{V,E}\}$ is a graph formed with a subset of nodes $\mathcal{V'}\subset\mathcal{V}$ and a subset of edges $\mathcal{E'}\subset\mathcal{E}$. Furthermore, the subset $\mathcal{V}'$ must contain all of the nodes involved in the edges in the subset $\mathcal{E}'$.
## Connected Component
Given a graph $\mathcal{G} = \{\mathcal{V,E}\}$, a subgraph $\mathcal{G}' = \{\mathcal{V',E'}\}$ is said to be a connected component if there is at least one path between any pair of nodes in the subgraph and the nodes in $\mathcal{V}'$ are not adjacent to any nodes in $\mathcal{V}/\mathcal{V'}$.
## Connected Graph
A graph $\mathcal{G} = \{\mathcal{V,E}\}$ is said to be connected if it has exactly one component.
## Shortest Path
Given a pair of nodes $v_s,v_t\in\mathcal{V}$ in a graph $\mathcal{G}$, we denote the set of paths from node $v_s$ to $v_t$ as $\mathcal{P}_{st}.$The shortest path between node $v_s$ and node $v_t$ is defined as $$ p_{st}^{sp} = \arg \min_{p\in\mathcal{P}_{st}}|p|$$where $p$ denotes a path in $\mathcal{P}_{st}$ with $|p|$ its length and $p_{st}^{sp}$ indicates the shortest path. Note that there could be more than one shortest path between any given pair of nodes.

The shortest path between a pair of nodes describes important information between them. Collective information on the shortest paths between any pairs of nodes in a graph indicates important characteristics of the graph. Specifically, the diameter of a graph is defined as the length of the longest shortest path in the graph.
## Diameter
Given a connected graph $\mathcal{G} = \{\mathcal{V,E}\}$, its diameter is defined as follows:$$\text{diameter}(\mathcal{G}) = \max_{v_s,v_t\in\mathcal{V}}\min_{p\in\mathcal{P}_{st}}|p|.$$


