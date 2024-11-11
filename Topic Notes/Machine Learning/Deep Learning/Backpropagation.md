Backpropagation, short for "backward propagation of errors," is a fundamental algorithm used to train artificial neural networks. It plays a crucial role in enabling these networks to learn from data by adjusting their internal parameters (weights and biases) to minimize the difference between their predictions and the actual outcomes. 

The primary goal of backpropagation is to optimize the neural network's performance by minimizing a loss (or cost) function. The loss function quantifies the discrepancy between the network's predictions and the true target values. By minimizing the loss, the network becomes better at making accurate predictions.

Backpropagation consists of two pases:
1. *Forward Pass*: the inputs are fed into and pass through the neural networks and the outputs are calculated using the current set of parameters, which are then used to evalkuate the value of the loss function
2. *Backward Pass*: the goal is to calculate the gradients of the loss function with respect to the parameters. According to the chain rule, the gradients for all parameters can be calculated dynamically in a backward direction starting from the output layer. 

## Backward Pass

Given a sequence of connected neural units $h^0, h^1, \dots, h^k, o$ from different layers where $h^i$ denotes a unit from the $i^{th}$ layer with $h^0$ from the input layer and $o$ from the output layer. Assuming that this is the only path going through the edge $(h^{r-1}, h^r)$, we can calculate the derivative using the chain rule as follows:
$$
\frac{\partial L}{\partial w_{(h^{r-1}, h^r)}} = \frac{\partial L}{\partial o} ᐧ \left[\frac{\partial o}{\partial h^k}\prod_{i=r}^{k-1}\frac{\partial h^{i+1}}{\partial h^i}\right]ᐧ\;\frac{\partial h^r}{\partial w_{(h^{r-1}, h^r)}} \forall r ∈ 1\dots k,
$$
where $w(h^{r-1}, h^r)$ denotes the parameter between the neural units $h^{r-1}$ and $h^r$.

In multilayer neural networks, we often have several paths going through the edge $(h^{r-1}, h^r)$. Hence, we need to sum up the gradients calculated through different paths as follows:
$$
\frac{\partial L}{\partial w_{(h^{r-1}, h^r)}} = \underbrace{\frac{\partial L}{\partial o} ᐧ \left[\sum_{[h^r, h^{r+1},\dots,h^k,o]∈\mathcal{P}}\frac{\partial o}{\partial h^k}\prod_{i=r}^{k-1}\frac{\partial h^{i+1}}{\partial h^i}\right]}_{\text{Backpropagation computes }\Delta(h^r,o)=\frac{\partial L}{\partial h^r}}ᐧ\;\frac{\partial h^r}{\partial w_{(h^{r-1}, h^r)}},
$$
where $\mathcal{P}$ denotes the set of paths starting from $h^r$ to $o$, which can be extended to pass the edge $(h^{r-1},h^r)$. There are two parts on the right hand side, where the second part is trouble-free to calculate and the first part (annotated as $\Delta(h^r,0)$) can be calculated recursively. Next we discuss how to recursively evaluate the first term. Specifically, we have

$$
\begin{eqnarray}
\Delta(h^r,0) &=& \frac{\partial L}{\partial o} ᐧ \left[\sum_{[h^r, h^{r+1},\dots,h^k,o]∈\mathcal{P}}\frac{\partial o}{\partial h^k}\prod_{i=r}^{k-1}\frac{\partial h^{i+1}}{\partial h^i}\right] \\
&=& \frac{\partial L}{\partial o} ᐧ \left[\sum_{[h^r, h^{r+1},\dots,h^k,o]∈ \mathcal{P}}\frac{\partial o}{\partial h^k}\prod_{i=r+1}^{k-1}\frac{\partial h^{i+1}}{\partial h^i}\;\dot\;\frac{\partial h^{r+1}}{\partial h^r}\right].
\end{eqnarray}
$$
We can decompose any path $P ∈ \mathcal{P}$ into two parts: the edge $(h^r,h^{r+1})$ and the remaining path from $h^{r+1}$ to $o$. Then, we can categorize the paths in $\mathcal{P}$ using the edge $(h^r,h^{r+1})$. Specifically, we denote the set of paths in $\mathcal{P}$ that share the same edge $(h^r,h^{r+1})$ as $\mathcal{P}_{r+1}$. Because all paths in $\mathcal{P}_{r+1}$ share the same first edge $(h^r,h^{r+1})$, any path in $\mathcal{P}_{r+1}$ can be characterized by the remaining path (i.e., the path from $h^{r+1}$ to $o$) besides the first edge. We denote the set of remaining paths as $\mathcal{P}_{r+1}'$. Then, we can continue to simply as follows:
$$
\begin{eqnarray}
\Delta(h^r,0) &=& \frac{\partial L}{\partial o} ᐧ \left[\sum_{(h^r,h^{r+1}\in\mathcal{E})}\frac{\partial h^{r+1}}{\partial h^r}ᐧ\left[\sum_{[h^{r+1},\dots,h^k,o]∈\mathcal{P}'_{r+1}}\frac{\partial o}{\partial h^k}\prod_{i=r+1}^{k-1}\frac{\partial h^{i+1}}{\partial h^i}\right]\right] \\
&=& \sum_{(h^r,h^{r+1}\in\mathcal{E})}\frac{\partial h^{r+1}}{\partial h^r}ᐧ\frac{\partial L}{\partial o}ᐧ\left[\sum_{[h^{r+1},\dots,h^k,o]∈\mathcal{P}'_{r+1}}\frac{\partial o}{\partial h^k}\prod_{i=r+1}^{k-1}\frac{\partial h^{i+1}}{\partial h^i}\right]
\end{eqnarray}
$$
