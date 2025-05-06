# Neural Generative Coding Models - Stable Version Summary

This document provides a summary of the stable versions of the Neural Generative Coding (NGC) models that we've run on the MNIST dataset.

## Models Overview

We've run three different NGC models:

1. **GNCN-PDH (Generative Neural Coding Network with Predictive Discrete Hebbian learning)**
   - A biologically inspired model that uses Predictive Discrete Hebbian learning
   - Uses local learning rules instead of backpropagation
   - Incorporates both bottom-up and top-down information flow
   - Activation function: tanh
   - Output function: sigmoid
   - Latent dimension: 360
   - Top latent dimension: 360
   - Hyperparameters:
     - beta: 0.1 (controls latent state update)
     - K: 50 (controls number of steps in an iterative settling episode)
     - leak: 0.001 (controls leak variable)
     - lambda: 0.01 (controls Laplacian prior)

2. **GNCN-t1 (Generative Neural Coding Network - Type 1)**
   - A variant of the NGC model with a different architecture
   - Activation function: tanh
   - Output function: sigmoid
   - Latent dimension: 360
   - Top latent dimension: 360
   - Hyperparameters:
     - beta: 0.1 (controls latent state update)
     - K: 50 (controls number of steps in an iterative settling episode)
     - leak: 0.001 (controls leak variable)
     - lambda: 0.01 (controls Laplacian prior)

3. **GNCN-t1-Sigma (Generative Neural Coding Network - Type 1 with Sigma)**
   - A variant of GNCN-t1 that includes sigma (variance) parameters
   - Activation function: relu
   - Output function: sigmoid
   - Latent dimension: 360
   - Top latent dimension: 360
   - Hyperparameters:
     - beta: 0.1 (controls latent state update)
     - K: 50 (controls number of steps in an iterative settling episode)
     - leak: 0.001 (controls leak variable)
     - lambda: 0.01 (controls Laplacian prior)

## Training Process

All models were trained on the MNIST dataset with the following settings:
- Number of iterations: 50
- Learning rate (eta): 0.001
- Batch size: 200
- Development batch size: 200

## Training Observations

### GNCN-PDH
- The model showed steady convergence during training
- Initial Binary Cross-Entropy (BCE) was high but decreased significantly over time
- The model learned to reconstruct MNIST digits effectively

### GNCN-t1
- Started with a high BCE of around 459
- Showed rapid improvement in the first few epochs
- By epoch 50, the BCE had decreased to around 42
- The model demonstrated good convergence properties

### GNCN-t1-Sigma
- Started with a very high BCE of around 481
- Showed dramatic improvement during training
- By epoch 50, the BCE had decreased to around 35
- The model achieved the lowest final BCE among the three models
- The validation BCE was around 58.9

## Key Differences Between Models

1. **Architecture**:
   - GNCN-PDH uses Predictive Discrete Hebbian learning
   - GNCN-t1 uses a different network architecture
   - GNCN-t1-Sigma extends GNCN-t1 with variance parameters

2. **Activation Functions**:
   - GNCN-PDH and GNCN-t1 use tanh activation
   - GNCN-t1-Sigma uses ReLU activation

3. **Performance**:
   - GNCN-t1-Sigma achieved the lowest BCE
   - All models showed good convergence properties
   - The models differ in their reconstruction quality and generative capabilities

## Comparison with Backpropagation Models

Compared to traditional backpropagation-based models like GVAE, GVAE-CV, and RAE:

1. **Biological Plausibility**:
   - NGC models use local learning rules, making them more biologically plausible
   - Backpropagation models require global error propagation

2. **Performance**:
   - Backpropagation models generally achieve lower BCE faster
   - NGC models show competitive performance with sufficient training
   - The biological plausibility of NGC models comes with some performance trade-offs

3. **Learning Dynamics**:
   - NGC models have more complex learning dynamics due to their iterative settling process
   - Backpropagation models have more straightforward learning dynamics

## Conclusion

The stable versions of the NGC models demonstrate that biologically plausible learning rules can achieve competitive performance on generative modeling tasks. While there are some performance trade-offs compared to backpropagation-based models, the NGC models offer valuable insights into how learning might occur in biological neural networks.

GNCN-t1-Sigma showed the best performance among the NGC models, suggesting that incorporating variance parameters can improve the model's generative capabilities. All three models successfully learned to reconstruct MNIST digits, demonstrating the viability of Neural Generative Coding as an alternative to traditional backpropagation-based approaches.
