A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed for processing structured grid-like data, such as images or time-series data. CNNs are particularly well-suited for tasks involving spatial hierarchies and local patterns, such as object detection, image classifications, and natural language processing.

Convolutional layers mainly have three key properties:
1. *sparse connections*
2. *parameter sharing*
3. *equivariant representation*
### Sparse Connections

In traditional neural network layers, the interactions between the input units and the output units can be described by a matrix. Each element of this matrix defines an independent parameter for the interaction between each input unit and each output unit. However, the convolutional layers usually have sparse connections between layers when the kernel is only nonzero on a limited number of input units. In densely connected layers, a single output unit is affected by all of the input units, whereas in the convolutional  neural network layers, the output unit is only affected by input units within the scope of the kernel, called the *receptive field*.

One of the major advantages of the sparse connectivity is that it can largely improve the computational efficiency. If there are $N$ input units and $M$ output units, there $N\times M$ parameters in the traditional neural network layers. The time complexity for a single computation pass of this layer is $O(N \times M)$. The convolutional layers with the same number of input and output units only have $K \times M$ parameters, when its kernel size is $K$. Typically, the kernel size $K$ is much smaller than the number of input units $N$.
### Parameter Sharing

Parameter sharing means sharing the same set of parameters when performing the calculation for different output units. In the convolutional layers, the same kernel is used to calculate the values of all the output units, which naturally leads to parameter sharing. In general, for convolutional layers with a kernel size $K$, there are $K$ parameters. Comparing with $N \times M$ parameters in the traditional neural network layers, $K$ is much smaller and consequently the requirement for memory is much lower.
### Equivariant Representation

A function is said to be *equivariant* if the output changes in the same way as the input changes. More specifically, a function $f$ is equivariant to another function $g$ if $f(g(x)) = g(f(x))$. In the case of the [[Convolution Operation]], it is not difficult to verify that it is equivariant to translation functions such as shifts. For example if we shift the input units to the right by one unit, we can still find the same output pattern that is also shifted to the right by one unit. This property is important in many applications where we care more about whether a certain feature appears than where it appears. For example, when recognizing whether an image is of a cat or not, we care whether there are some important features in the image indicating the existence of a cat instead of where these features show in the image. The property of equivariant to translation of CNNs is crucial to their success in the area of image classification.
## Convolutional Layers in Practice

Generally, the $i^{th}$ channel of the input consists of the $i^{th}$ element of vectors at all positions of the input. The length of the vector at each position (e.g., the pixel in the case of images) is the number of channels. Hence, the convolution typically involves three dimensions, though it only "slides" in two dimensions (it does not usually slide in the dimension of channels). In typical convolutional layers, multiple distinct kernels are applied in parallel to extract features from the input layer. As a consequence, the output layer is also multichannel where the results for each kernel correspond to each output channel. Let us consider an input image $I$ with $L$ channels. The convolution operation with $P$ kernels can be formulated as $$S(i,j,p) = (I*K_p)(i,j) = \sum_{l=1}^L\sum_{\tau=i-n}^{i+n}\sum_{j=\gamma-n}^{\gamma+n}I(\tau,\gamma,l)K_p(i-\tau,j-\gamma,l),$$where $K_p$ is the $p^{th}$ kernel with $(2n+1)^2\cdot L$ parameters. The output consists of $P$ channels.

In many cases, to further reduce computational complexity, we can regularly skip some positions when sliding the kernel over the input. The convolution can be only performed every $s$ positions, where $s$ is called the *stride*. We call the convolutions with stride *strided convolutions*. The strided convolution can also be viewed as a downsampling over the results of the regular convolution. Zero padding is usually applied to the input to maintain the size of the output. The size of the padding, the size of the receptive field (or the size of the kernel), and the stride determine the size of the output when the input size is fixed. Given a one-dimensional input with size $N$, a padding size of $Q$, a size of the receptive field $F$, and a stride size $s$, the size of the output $O$ can be calculated as $$O=\frac{N-F+2Q}{s}+1.$$
## Pooling/Downsampling

A pooling layer usually follows the convolution layer and the detector (or activation) layer. The pooling function summarizes the statistic of a local neighborhood to denote this neighborhood in the resulting output. Hence, the width and height of the data is reduced after the pooling layer. However, the depth (number of channels) does not change. The commonly used pooling operations are either max pooling or average pooling.

Pooling is a powerful operation that frees filters from requiring their inputs to be in precisely the right place. Mathematicians refer to a change in location as *translation* or *shift*, and if some operation is insensitive to a certain kind of change it's called *invariant* with respect to that operation. Combining these, we sometimes say that pooling allows our convolutions to be *translationally invariant*, or *shift invariant*. Pooling also also has the bonus benefit of reducing the size of the tensors flowing through our network, which reduces both memory needs and execution time.
## Striding

Striding refers to the step size or interval by which the convolution filter (or kernel) moves across the input feature map. It controls how much the filter shifts during each operation, determining the spatial dimensions of the output feature map. Larger strides lead to fewer computations and smaller feature maps. Smaller strides capture more overlapping information, which can be useful for detailed feature extraction.
## Example

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(32 * 8 * 8, 128)
		self.fc2 = nn.Linear(128, 10) # 10 output classes

	def forward(self, x):
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = x.view(-1, 32 * 8 * 8)
		x = torch.relu(self.fc1(x))
		x = self.fx2(x)
		return x

model = SimpleCNN()
```