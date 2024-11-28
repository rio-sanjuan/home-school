# t-Distributed Stochastic Neighbor Embedding (t-SNE)

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a non-linear dimensionality reduction technique well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.

### Key Components of t-SNE:

1. **Probability Distributions in Original Space:** t-SNE starts by converting the pairwise Euclidean distances between points into conditional probabilities that represent similarities. The similarity of data point \( x*j \) to data point \( x_i \) is the conditional probability, \( p*{j|i} \), that \( x_i \) would pick \( x_j \) as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at \( x_i \).

$$ p*{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma^2)}{\sum*{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma^2)} $$

2. **Probability Distributions in Reduced Dimensional Space:** In the lower-dimensional space, a similar probability distribution is computed, but with a crucial difference: it uses a Student’s t-distribution with one degree of freedom (Cauchy distribution) instead of a Gaussian distribution to compute the similarity between two points.

$$
q*{j|i} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum*{k \neq i} (1 + ||y_i - y_k||^2)^{-1}}
$$

where \( $y_i$ \) and \( $y_j$ \) are mappings of \( $x_i$ \) and \( $x_j$ \) in the lower-dimensional space.

3. **Optimization (Gradient Descent):** t-SNE aims to minimize the divergence between the two distributions \( P \) (in the high-dimensional space) and \( Q \) (in the low-dimensional space) using a cost function, the Kullback-Leibler (KL) divergence:

$$
C = \sum*i \text{KL}(P_i || Q_i) = \sum_i \sum_j p*{j|i} \log \frac{p*{j|i}}{q*{j|i}}
$$

It employs a gradient descent method to minimize the cost function.

### Important Characteristics of t-SNE:

- **Preservation of Local Structures:** t-SNE is particularly known for its ability to preserve local structures within the data.

- **Crowding Problem:** It alleviates the crowding problem (the problem of mapping high-dimensional data that is distributed in a complex manner to lower-dimensional space) by utilizing a t-distribution in the reduced space, which has heavier tails than the Gaussian used in the original space.

- **Computational Complexity:** The algorithm can be computationally intensive, especially for large datasets, due to the pairwise computations of distances and similarities.

- **Perplexity:** A crucial hyperparameter to be set in t-SNE is perplexity, which loosely determines how to balance attention between preserving local and global aspects of the data. Typical values for perplexity range between 5 and 50.

- **Randomness:** t-SNE involves some randomness (due to the initial configuration of points in the low-dimensional space), and therefore different runs can yield different results.

### Practical Implications:

- **Data Visualization:** t-SNE is extensively utilized in visualizing high-dimensional data in various fields like genomics, document analysis, and image processing.

- **Cluster Identification:** It is also used for identifying clusters within the data, given its ability to segregate different types of data points in the visualization clearly.

Understanding t-SNE’s mechanisms, when to use it, and how to interpret its outputs are fundamental when heading into data-related fields, as it is widely used and discussed in the context of exploratory data analysis and representation learning.
