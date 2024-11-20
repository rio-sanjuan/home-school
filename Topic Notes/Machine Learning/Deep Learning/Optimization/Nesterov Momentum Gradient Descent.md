The key idea behind Nesterov Momentum is that instead of using only the gradient at the location where we currently are, we also use the gradient at the location where we expect that we're going to be. Because we can't really predict the future, we estimate where we're going to be on the next step and use the gradient there. The thinking is that if the error surface is relatively smooth, and our 