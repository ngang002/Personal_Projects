# Stock Predictions

In this project I attempt to predict the path of a stock N days after some established time period.  This model implements the use of a Long-Short Term Memory (LSTM) Network model, which allows for recurrent updating on the neural network.  Recurrent Neural Networks (RNN) are ideal for dealing with time series datasets, as a previous iteration will inform future iterations, and these effects will propogate through the data.  In this case we use the examples of stocks, as yesterday's ticker could largely impact current price, especially if for example a company undergoes a major product rollout, or is undergoing bankrupcy.


RNNs are intuitively neural networks implement the gradients in the loss, and then using back-propogation to update the weights.  However, in doing so we are likely to encroach on the Vanishing Gradient Problem.  The Vanishing Gradient Problem is defined as a problem in which the changes to the loss function are minimal as we update the weights, and we will likely be stuck in a local minima without truly exploring the full parameter space of allowed weights.  In doing so, we are stuck, and the changes to the weights and biases of the nodes will be too small and prevent us from approaching those parameters which are optimal.  


A LSTM is ideal for dealing with the Vanishing Gradient Problem, as some nodes will "forget" (with the implementation of a "forget gate"), while remembering 
