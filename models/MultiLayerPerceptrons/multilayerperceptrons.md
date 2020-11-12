# Multi-Layer Perceptrons

![Multi-Layer Perceptron architecture](https://i.ibb.co/thCPLXS/image.png)
Fig 1: A Multi-Layer Perceptron

### What are Multi-Layer Perceptrons?

Multi-Layer Perceptrons (MLPs) are a type of feed-forward artificial neural networks (a neural network where nodes don't form a cycle) that are comprised of at least three layers of neurons. Each layer of neurons is connected to every neuron in the next layer by "weights" (and sometimes "biases"). Because of the connectivity of the layers, the weights can be represented by a matrix (and biases can be represented by a vector).

If weights are represented by matrices, biases are represented by vectors, and the inputs are represented by vectors, we can compute each successive layer as a matrix multiplication of the weights and inputs followed by a vector summation with the biases.

For a ![4](http://latex.codecogs.com/svg.latex?4)-dimensional column vector input ![\vec{x}](http://latex.codecogs.com/svg.latex?\vec{x}), ![4](http://latex.codecogs.com/svg.latex?4)-dimensional column vector bias ![\vec{b}](http://latex.codecogs.com/svg.latex?\vec{b}), and a ![3\times4](http://latex.codecogs.com/svg.latex?3\times4) weights matrix ![\mathbf{W}](http://latex.codecogs.com/svg.latex?\mathbf{W}), the computation from one layer to the next can be done as such:

![\vec{o}=\mathbf{W}\vec{x}+\vec{b}](http://latex.codecogs.com/svg.latex?\vec{o}=\mathbf{W}\vec{x}+\vec{b}), where ![\vec{o}](http://latex.codecogs.com/svg.latex?\vec{o}) is the output.

However, this is not the only step necessary to creating a successful model. If we only had a series of matrix multiplications, there would be no advantage over simply training with a just two layer (input and output), because matrices act linearly upon vectors.

To fix this, we use activation functions!

![](https://mlfromscratch.com/content/images/size/w2000/2019/12/activation-functions.gif)

Sourced from: [https://mlfromscratch.com/](https://mlfromscratch.com/)

Activation functions are functions that are applied to the matrix multiplication and summation for each layer to create the inputs for the next layer. The provide some non-linearity to the model, which allows for deeper models to be more effective. The activation function was inspired by biological neurons, which only get activated after a threshold input.

After adding the activation function ![(f(x))](http://latex.codecogs.com/svg.latex?(f(x))), the computation from one layer to the next can be written as:

![\vec{y}=f(\vec{o})](http://latex.codecogs.com/svg.latex?\vec{y}=f(\vec{o})), where ![\vec{y}](http://latex.codecogs.com/svg.latex?\vec{y}) is the input to the next layer.

Now that we've covered how to create a feed-forward MLP, how do we actually train it?

### Training

Usually, when training MLPs, we are doing what is known as "supervised learning." Supervised learning is when you train a model on a dataset with input as well as the corresponding expected outputs. On the other hand, we have "unsupervised learning," in which the model has to discover patterns in the data completely on its own. Some combinations of the two also exist, giving rise to "semi-supervised learning."

When we feed an input to the model, it will perform a series of calculations that produce an output that may or may not be accurate. Given a certain configuration pulled from its parameter space (a vector space in ![n](http://latex.codecogs.com/svg.latex?n) dimensions, where ![n](http://latex.codecogs.com/svg.latex?n) is the parameter count), the model with produce a certain output. The goal of training is to find the point in the parameter space with the most accurate answer.

Sure, the concept sounds simple, but how do we actually compare the accuracy of our model? Well, the answer is to use something called a loss function.

<font color="red">Note on notation</font>: the configuration of the parameters is represented by a vector (often called ![\theta](http://latex.codecogs.com/svg.latex?\theta)), which is also an input to the model "function."

#### Loss Functions

As you can probably guess from its name, a loss function is a function that tells you how badly your model is doing.

For our model ![M](http://latex.codecogs.com/svg.latex?M), parameters ![\theta](http://latex.codecogs.com/svg.latex?\theta), and training input ![\vec{x}](http://latex.codecogs.com/svg.latex?\vec{x}) and output ![\vec{y}](http://latex.codecogs.com/svg.latex?\vec{y}), we can compare the output of the model ![M(\vec{x},\theta)](http://latex.codecogs.com/svg.latex?M(\vec{x},\theta)) to the target output ![\vec{y}](http://latex.codecogs.com/svg.latex?\vec{y}) with a loss function ![\mathcal{L}(\vec{x_0},\vec{x_1})](http://latex.codecogs.com/svg.latex?\mathcal{L}(\vec{x_0},\vec{x_1})).

Our loss: ![\mathcal{L}(M(\vec{x},\theta),\vec{y})](http://latex.codecogs.com/svg.latex?\mathcal{L}(M(\vec{x},\theta),\vec{y}))

Some commonly used loss functions include:

Mean-Squared Error: ![\mathcal{L}_{MSE}(\vec{x_0},\vec{x_1})=(\vec{x_0}-\vec{x_1})^2](http://latex.codecogs.com/svg.latex?\mathcal{L}_{MSE}(\vec{x_0},\vec{x_1})=(\vec{x_0}-\vec{x_1})^2)

Mean-Absolute Error: ![\mathcal{L}_{MAE}(\vec{x_0},\vec{x_1})=(\vec{x_0}-\vec{x_1})^2](http://latex.codecogs.com/svg.latex?\mathcal{L}_{MAE}(\vec{x_0},\vec{x_1})=|\vec{x_0}-\vec{x_1}|)

Binary Cross-Entropy: ![\mathcal{L}_{BCE}(\vec{x_0},\vec{x_1})=-(\vec{x_1}\log(\vec{x_0})+(1-\vec{x_1})\log(1-\vec{x_0}))](http://latex.codecogs.com/svg.latex?\mathcal{L}_{BCE}(\vec{x_0},\vec{x_1})=-(\vec{x_1}\log(\vec{x_0})+(1-\vec{x_1})\log(1-\vec{x_0})))

By taking advantage of some very neat calculus, we can update the parameter configuration ![\theta](http://latex.codecogs.com/svg.latex?\theta) to progressively improve the accuracy of the model.

#### Backpropagation

If you can remember multi-variable calculus, we learned how to use the gradient ![(\nabla f)](http://latex.codecogs.com/svg.latex?(\nabla%20f)) to find the direction in which the value of a function increases at the greatest rate. Because the gradient gives us the direction of the greatest rate of change, the negative of the gradient will give us the direction in which the value of a function decreases the most.

Before we continue, let's define some notation. In this document, unless stated otherwise:

![(\nabla_\mathbf{x}f)_i=\frac{\partial f}{\partial\mathbf{x}_i}](http://latex.codecogs.com/svg.latex?(\nabla_\mathbf{x}f)_i=\frac{\partial%20f}{\partial\mathbf{x}_i})

given that ![\mathbf{x}](http://latex.codecogs.com/svg.latex?\mathbf{x}_i) is a vector of all the variable we are taking the derivative with respect to.

Now, with that, if we wanted to find the direction in which to tweak the model parameters, all we would have to do is take the negative gradient of the loss with respect to the model parameters ![(-\nabla_\theta\mathcal{L})](http://latex.codecogs.com/svg.latex?(-\nabla_\theta\mathcal{L})). The reason we don't take the gradient with respect to the input is because we are trying to determine how the model parameters affect the output rather than the input.

We now know the math behind optimizing the parameters, but how do we actually calculate the gradient of the loss? Luckily, the handy ol' chain rule is perfect for this problem!

Let's say that we're using the Mean Square Error loss. To take the gradient:

![-\nabla_\theta\mathcal{L}_{MSE}(M(\vec{x},\theta),\vec{y})](http://latex.codecogs.com/svg.latex?-\nabla_\theta\mathcal{L}_{MSE}(M(\vec{x},\theta),\vec{y}))

![\newline=-\nabla_\theta(M(\vec{x},\theta)-\vec{y})^2\newline=-2(M(\vec{x},\theta)-\vec{y})\nabla_\theta(M(\vec{x},\theta)-\vec{y})\newline=-2(M(\vec{x},\theta)-\vec{y})\nabla_\theta{M}(\vec{x},\theta)](http://latex.codecogs.com/svg.latex?\newline=-\nabla_\theta(M(\vec{x},\theta)-\vec{y})^2\newline=-2(M(\vec{x},\theta)-\vec{y})\nabla_\theta(M(\vec{x},\theta)-\vec{y})\newline=-2(M(\vec{x},\theta)-\vec{y})\nabla_\theta{M}(\vec{x},\theta))

Hooray, we've managed to rewrite the gradient of the loss in terms of the gradient of the model itself. But how do we go further?

To do this, we take advantage of computational trees. In the computational tree pictured below, we can see that the "leaves" of our graph are variables that are operated upon by these nodes. If we wanted to take the derivative of with respect to any node, we can take advantage of the chain rule.

![Computational Tree](https://i.ibb.co/ZfY5f6b/ezgif-2-aa5000065129.gif)
