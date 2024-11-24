Another [[Regularization]] technique is called *batch normalization*, often referred to simply as batchnorm. Like [[Dropout]], batchnorm can be implemented as a layer without neurons. Unlike dropout, batchnorm actually does perform some computation, though there are no parameters for us to specify. Batchnorm modifies the values that come out of a layer. 

Recall that many activation functions, like [[Leaky ReLU]] and [[Hyperbolic Tangent]], have their greatest effect near 0. To get the most benefit from those functions, we need the numbers flowing into them to be in a small range centered around 0. That's what batchnorm does by scaling and shifting all the outputs of a layer together. Because batchnorm moves the neuron outputs into a small range near 0, we're less prone to seeing any neuron learning one specific detail and producing a huge output that swamps all the other neurons, and thus we are able to delay the onset of #overfitting. Batchnorm scales and shifts all the values coming out of the previous layer over the course of an entire mini-batch in just this way. It learns the parameters for this scaling and shifting along with the weights in the network so they take on the most useful values.

We apply batchnorm before the activation function so that the modified values will fall in the region of the activation function where they are affected the most. In practice, this means we place no activation function on the neurons going into batchnorm (or if we must specify a function, it's the [[Linear Activation Function]]). Those values go into batchnorm, and then they're fed into the activation function we want to apply.