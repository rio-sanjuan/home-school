## Adaptive Moment Estimation

ADAM combines the ideas of momentum and adaptive learning rates. It computes an adaptive learning rate for each parameter by using estimates of the first and second moments (mean and variance) of the gradients.

## Momentum Gradient Descent

Momentum helps accelerate convergence by taking into account the past gradients, reducing oscillations and speeding up the search in the relevant direction. For each step, once we calculate how much we want each weight to change, we add in a small amount of its change from the previous step. If the change on a given step is 0, or nearly 0, but we had some larger change on the last step, we use some of that prior motion now, which pushes us along over the plateau.

We multiply the momentum, $m$, by a scaling factor usually referred to with the lowercase Greek letter $\gamma$. Sometimes, this is called the *momentum scaling factor*, and it's a value from 0 to 1. On each step, we first find the gradient and multiply it by the current value of the learning rate $\eta$, as before. Then we find the previous change, and scale it by $\gamma$, and add both of those changes to the current position of the weight. If we use too much momentum, our descent can fly right up the other side and out of the bowl altogether, but if we use too little momentum, our descent may not get across the plateaus it encounters along the way.

## Nesterov Momentum Gradient Descent

The key idea behind Nesterov Momentum is that instead of using only the gradient at the location where we currently are, we also use the gradient at the location where we expect that we're going to be. Because we can't really predict the future, we estimate where we're going to be on the next step and use the gradient there. The thinking is that if the error surface is relatively smooth, and our 

## RMSprop

RMSprop adjusts the learning rate based on a moving average of the squared gradients, helping with the problem of vanishing or exploding gradients.