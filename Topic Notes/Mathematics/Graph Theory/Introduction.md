Graphs, which describe pairwise relations between entities, are essential representations for real-world data from many different domains, including social science, linguistics, chemistry, biology, and physics. 

## Definitions

### Degree

In a graph $\mathcal{G} = \{\mathcal{V,E}\}$, the degree of a node $v_i\in\mathcal{V}$ is the number of nodes that are adjacent to $v_i$: $$ d(v_i) = \sum_{v_j\in\mathcal{V}}\mathbb{1}_\mathcal{E}(\{v_i, v_j\})$$ where $\mathbb{1}_\mathcal{E}(\cdot)$ is an indicator function: $$ \mathbb{1}_\mathcal{E}(\{v_i, v_j\}) = \begin{cases} 1 & \text{if } (v_i,v_j)\in\mathcal{E} \\ 0 & \text{if } (v_i,v_j)\notin\mathcal{E}\end{cases} $$