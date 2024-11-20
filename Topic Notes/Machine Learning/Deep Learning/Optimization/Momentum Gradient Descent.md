Momentum helps accelerate convergence by taking into account the past gradients, reducing oscillations and speeding up the search in the relevant direction. For each step, once we calculate how much we want each weight to change, we add in a small amount of its change from the previous step. If the change on a given step is 0, or nearly 0, but we had some larger change on the last step, we use some of that prior motion now, which pushes us along over the plateau.

We multiply the momentum, $m$, by a scaling factor usually referred to with the lowercase Greek letter $\gamma$. Sometimes, this is called the *momentum scaling factor*, and it's a value from 0 to 1. On each step, we first find the gradient and multiply it by the current value of the learning rate $\eta$, as before. Then we find the previous change, and scale it by $\gamma$, and add both of those changes to the current position of the weight. If we use too much momentum, our descent can fly right up the other side and out of the bowl altogether, but if we use too little momentum, our descent may not get across the plateaus it encounters along the way.