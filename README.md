# Deep-Neural-Network-Implementation
My implementation of some basic Neural Networks, ideas from papers, and some of my own ideas. (Mostly done in TensorFlow). 
(All of the code are run on NVIDIA TITAN X (Pascal) single GPU, and reported runtime)

## Basic Implementation
Basic implementation of MLP, CNN, RNN are in basic directory. 

## Implementation of models from research papers
In paper directory, but with some small changes because can't get exact same data as the paper, and computation power. 
  1. AlexNet
  2. LeNet-5

## Some Crazy Ideas:
Feel free to clone and try out those ideas! 

### Neural Network Architure: 
  1. Some weird way to link neurons, like the current layer get input from all layers before. A full version of ResNet that each later take all values from all previous layers. Potentially add some convolution layers to reduce the parameters. An implementation with TensorFlow is [here](https://github.com/shansixiong/Deep-Neural-Network-Implementation/tree/master/random_ideas/FNN). 

  2. Random structured neurons, linked in weird ways

  3. No backprop, because neurons dont give info back to neurons before


### Optimization:
  #### Exploding/Vanishing weight: 
  
  1. Different learning rate for different layer to address the explosion increase. 
  Lower learning rate for eariler layers to make them less subjected to the back prop derivative. 
  2. Transfer Learning: train one layer on part of data first, and add a layer before it, 
  and recursively do this. To make earlier layers train on less iterations. 


### Application:
  #### Sequence Generation: 
  1. Feed art pictures to some RNN, and let it draw pictures. 
  2. Compress data, and a DL learn to match the compressed to orginal
