# Neural Generative Coding (NGC): A Comprehensive Guide

## Introduction

Neural Generative Coding (NGC) is a biologically inspired approach to neural networks based on the predictive coding theory from neuroscience. This document provides a comprehensive explanation of NGC models, their mathematical foundations, and their relationship to other generative models like Variational Autoencoders (VAEs) and Regularized Autoencoders (RAEs).

## Table of Contents
1. [Theoretical Framework](#1-theoretical-framework)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [NGC Model Variants](#3-ngc-model-variants)
4. [Comparison with Other Models](#4-comparison-with-other-models)
5. [Metaphorical Examples](#5-metaphorical-examples)
6. [Computational Implementation](#6-computational-implementation)
7. [Practical Applications](#7-practical-applications)
8. [Future Directions](#8-future-directions)

## 1. Theoretical Framework

### 1.1 Predictive Coding Theory

Predictive coding is a theory from neuroscience that posits that the brain is constantly trying to predict its sensory inputs and updates its internal models based on prediction errors. According to this theory:

- The brain has a hierarchical structure where higher levels predict the activity of lower levels
- The difference between predicted and actual activity generates prediction errors
- These prediction errors drive learning and updating of internal models
- Information flows both top-down (predictions) and bottom-up (prediction errors)

### 1.2 Neural Generative Coding

Neural Generative Coding (NGC) is a computational implementation of predictive coding principles. In NGC:

- The network consists of multiple layers, each trying to predict the activity of the layer below it
- Prediction errors drive the learning process through local learning rules
- The learning rules are Hebbian-like, making them more biologically plausible than backpropagation
- The network involves an iterative settling process where it converges to a stable state before weight updates are applied

## 2. Mathematical Formulation

### 2.1 Basic NGC Framework

Consider a hierarchical generative model with $L$ layers. For each layer $l$, we have:

- State variables $\mathbf{s}^l$: Represent the activity of neurons in layer $l$
- Prediction of the layer below $\hat{\mathbf{s}}^{l-1} = f^l(\mathbf{s}^l)$: Layer $l$ predicts the activity of layer $l-1$
- Prediction error $\mathbf{e}^{l-1} = \mathbf{s}^{l-1} - \hat{\mathbf{s}}^{l-1}$: The difference between actual and predicted activity

### 2.2 State Update Rule

The state update rule for layer $l$ is:

$$\Delta \mathbf{s}^l = \eta_s \left( \mathbf{W}^l \mathbf{e}^{l-1} - \mathbf{e}^l \right)$$

Where:
- $\eta_s$ is the state update learning rate
- $\mathbf{W}^l$ is the weight matrix for layer $l$
- $\mathbf{e}^{l-1}$ is the prediction error from the layer below
- $\mathbf{e}^l$ is the prediction error from the layer above

This rule updates the state variables based on a combination of bottom-up and top-down prediction errors.

### 2.3 Weight Update Rule

The weight update rule follows a Hebbian-like learning rule:

$$\Delta \mathbf{W}^l = \eta_W \mathbf{e}^{l-1} (\mathbf{s}^l)^T - \lambda \mathbf{W}^l$$

Where:
- $\eta_W$ is the weight update learning rate
- $\lambda$ is a weight decay parameter for regularization

This rule strengthens connections between neurons that contribute to reducing prediction errors.

### 2.4 Iterative Settling Process

The iterative settling process in NGC involves:

1. Initialize state variables $\mathbf{s}^l$ for all layers
2. For $K$ steps:
   a. Compute predictions $\hat{\mathbf{s}}^{l-1} = f^l(\mathbf{s}^l)$ for all layers
   b. Compute prediction errors $\mathbf{e}^{l-1} = \mathbf{s}^{l-1} - \hat{\mathbf{s}}^{l-1}$ for all layers
   c. Update state variables $\mathbf{s}^l = \mathbf{s}^l + \Delta \mathbf{s}^l$ for all layers
3. Update weights $\mathbf{W}^l = \mathbf{W}^l + \Delta \mathbf{W}^l$ for all layers

This process allows the network to reach a stable state before updating the weights.

## 3. NGC Model Variants

### 3.1 GNCN-PDH (Generative Neural Coding Network with Predictive Discrete Hebbian learning)

GNCN-PDH is an implementation of NGC that uses local Hebbian-like learning rules:

- Uses Predictive Discrete Hebbian learning
- Incorporates both bottom-up and top-down information flow
- Activation function: tanh
- Output function: sigmoid
- Hyperparameters:
  - beta: 0.1 (controls latent state update)
  - K: 50 (controls number of steps in an iterative settling episode)
  - leak: 0.001 (controls leak variable)
  - lambda: 0.01 (controls Laplacian prior)

### 3.2 GNCN-t1 (Generative Neural Coding Network - Type 1)

GNCN-t1 is a variant of NGC with a different architecture:

- The key difference is in how the layers are connected and how the predictions are computed
- Activation function: tanh
- Output function: sigmoid
- Same hyperparameters as GNCN-PDH

### 3.3 GNCN-t1-Sigma (Generative Neural Coding Network - Type 1 with Sigma)

GNCN-t1-Sigma extends GNCN-t1 by including variance parameters:

- Includes variance parameters (sigma) that capture uncertainty in predictions
- The prediction in GNCN-t1-Sigma is: $\hat{\mathbf{s}}^{l-1} = f^l(\mathbf{s}^l, \boldsymbol{\sigma}^l)$
- The state update rule is modified to include the variance:
  $$\Delta \mathbf{s}^l = \eta_s \left( \mathbf{W}^l \frac{\mathbf{e}^{l-1}}{(\boldsymbol{\sigma}^{l-1})^2} - \frac{\mathbf{e}^l}{(\boldsymbol{\sigma}^l)^2} \right)$$
- The weight update rule becomes:
  $$\Delta \mathbf{W}^l = \eta_W \frac{\mathbf{e}^{l-1}}{(\boldsymbol{\sigma}^{l-1})^2} (\mathbf{s}^l)^T - \lambda \mathbf{W}^l$$
- Activation function: relu
- Output function: sigmoid

## 4. Comparison with Other Models

### 4.1 NGC vs. Variational Autoencoders (VAEs)

| Aspect | NGC | VAE |
|--------|-----|-----|
| Learning Mechanism | Local learning rules based on prediction errors | Backpropagation through the network |
| Biological Plausibility | High (local learning rules) | Low (requires global error propagation) |
| Probabilistic Framework | Implicit through prediction errors | Explicit through variational inference |
| Generative Process | Through iterative settling | Through sampling from the prior |
| Computational Efficiency | Lower (requires iterative settling) | Higher (single forward pass for inference) |
| Performance | Competitive with sufficient training | Generally better for density estimation |

### 4.2 NGC vs. Regularized Autoencoders (RAEs)

| Aspect | NGC | RAE |
|--------|-----|-----|
| Learning Mechanism | Local learning rules based on prediction errors | Backpropagation through the network |
| Biological Plausibility | High (local learning rules) | Low (requires global error propagation) |
| Probabilistic Framework | Implicit through prediction errors | None (deterministic model) |
| Regularization | Through prediction errors and weight decay | Explicit regularization terms |
| Computational Efficiency | Lower (requires iterative settling) | Higher (single forward pass for inference) |
| Performance | Competitive for reconstruction | Generally better for pure reconstruction |

### 4.3 Learning Rules Comparison

| Learning Rule | Description | Biological Plausibility | Used in |
|---------------|-------------|-------------------------|---------|
| Backpropagation | Computes gradients of the loss function with respect to all parameters using the chain rule | Low (requires symmetric weights, storing activations, non-local computations) | VAEs, RAEs |
| Hebbian Learning | "Neurons that fire together, wire together" - strengthens connections between co-active neurons | High (local learning rule) | Basic component of NGC |
| Predictive Coding | Combines Hebbian learning with prediction errors | High (local learning rule) | NGC models |

## 5. Metaphorical Examples

### 5.1 Predictive Coding as Weather Forecasting

Imagine a weather forecaster who makes daily predictions about tomorrow's weather:

- The forecaster (higher-level brain region) makes a prediction about tomorrow's weather (sensory input)
- The next day, the actual weather (sensory input) is observed
- The difference between the prediction and the actual weather is the prediction error
- The forecaster updates their forecasting model based on this prediction error
- Over time, the forecaster becomes better at predicting the weather

In this metaphor:
- The forecaster's prediction is analogous to top-down predictions in predictive coding
- The actual weather observation is analogous to bottom-up sensory input
- The prediction error drives learning and model updating
- The forecaster's model becomes more accurate over time, just as the brain's internal models do

### 5.2 NGC as a Symphony Orchestra

Think of an NGC network as a symphony orchestra:

- The conductor (top layer) has a score (internal model) and directs the musicians (lower layers)
- Each section of the orchestra (layer) tries to predict what the sections below it should play
- If a section plays something different from what was expected (prediction error), this information flows up to the conductor
- The conductor adjusts their directions (top-down signals) based on these errors
- Over many rehearsals (training iterations), the orchestra learns to play in harmony

In this metaphor:
- The conductor's score represents the top-level internal model
- The musicians represent neurons in different layers
- The sound produced by each section represents the activity of each layer
- The discrepancies between expected and actual sounds represent prediction errors
- The iterative process of rehearsal represents the training process

### 5.3 GNCN-t1-Sigma as a Team of Estimators

Imagine a team of estimators trying to guess the number of marbles in jars:

- Each estimator (layer) makes a prediction about the number of marbles
- Each estimator also provides a confidence level (variance parameter) for their prediction
- When the actual number is revealed, the prediction errors are weighted by the confidence levels
- Estimators with higher confidence have more influence on the team's overall prediction
- Estimators with lower confidence have their errors weighted less heavily
- Over time, the team learns both better predictions and appropriate confidence levels

In this metaphor:
- The estimators represent different layers in the GNCN-t1-Sigma model
- The predictions represent the state variables
- The confidence levels represent the variance parameters (sigma)
- The weighted prediction errors represent the variance-scaled errors in GNCN-t1-Sigma
- The learning process adjusts both predictions and confidence levels

## 6. Computational Implementation

### 6.1 Basic NGC Algorithm

```
Initialize weights W and state variables s
For each training iteration:
    Present input x to the lowest layer
    For k = 1 to K (iterative settling):
        For each layer l from bottom to top:
            Compute prediction: s_hat[l-1] = f(s[l])
            Compute prediction error: e[l-1] = s[l-1] - s_hat[l-1]
        For each layer l from top to bottom:
            Update state: s[l] += eta_s * (W[l] * e[l-1] - e[l])
    For each layer l:
        Update weights: W[l] += eta_w * e[l-1] * s[l]^T - lambda * W[l]
```

### 6.2 GNCN-PDH Implementation

The GNCN-PDH model implements the basic NGC algorithm with specific activation functions and hyperparameters:

- Input layer: Raw data (e.g., MNIST images)
- Hidden layers: Use tanh activation
- Output layer: Uses sigmoid activation
- State update rate (eta_s): Controlled by beta parameter
- Weight update rate (eta_w): Learning rate parameter
- Iterative settling steps (K): Typically 50
- Weight decay (lambda): Typically 0.01

### 6.3 GNCN-t1-Sigma Implementation

The GNCN-t1-Sigma model extends GNCN-t1 with variance parameters:

- Each layer has both state variables and variance parameters
- Predictions are computed using both state variables and variance parameters
- Prediction errors are weighted by the inverse of the variance
- State updates and weight updates incorporate the variance-weighted errors
- Uses ReLU activation instead of tanh

## 7. Practical Applications

### 7.1 Neuroscience Research

NGC models provide a computational framework for testing hypotheses about predictive coding in the brain:

- Investigating how prediction errors might drive learning in biological neural networks
- Exploring the role of top-down and bottom-up information flow in perception
- Studying how the brain might represent uncertainty in its predictions

### 7.2 Generative Modeling

NGC models can be used for generative modeling tasks:

- Image generation: Learning to generate realistic images
- Anomaly detection: Identifying inputs that produce large prediction errors
- Data compression: Using the latent representations as compressed versions of the input

### 7.3 Biologically Inspired AI

NGC principles can inform the development of more biologically plausible AI systems:

- Neuromorphic computing: Implementing NGC-like algorithms on neuromorphic hardware
- Online learning: Developing systems that can learn continuously from streaming data
- Local learning rules: Creating more efficient learning algorithms that don't require backpropagation

## 8. Future Directions

### 8.1 Scaling to Complex Data

Current research is exploring how to scale NGC models to more complex data:

- Incorporating convolutional architectures for better image processing
- Adding recurrent connections for sequential data
- Developing hierarchical architectures for handling complex, structured data

### 8.2 Improving Efficiency

Efforts are underway to make NGC models more computationally efficient:

- Developing approximations to the iterative settling process
- Creating more efficient implementations of the learning rules
- Exploring hardware acceleration for NGC-specific computations

### 8.3 Hybrid Approaches

Researchers are investigating hybrid approaches that combine the strengths of NGC with other models:

- NGC-VAE hybrids that combine biological plausibility with probabilistic frameworks
- NGC models with attention mechanisms for better feature selection
- NGC principles applied to transformer architectures for improved scalability

### 8.4 Theoretical Advances

Ongoing theoretical work aims to deepen our understanding of NGC models:

- Analyzing the convergence properties of the iterative settling process
- Establishing connections to other frameworks like energy-based models
- Developing more rigorous probabilistic interpretations of NGC

## Conclusion

Neural Generative Coding represents a promising approach to neural networks that combines biological plausibility with competitive performance. By implementing principles from predictive coding theory, NGC models offer insights into how the brain might learn and process information, while also providing practical tools for generative modeling and other machine learning tasks.

The different variants of NGC models (GNCN-PDH, GNCN-t1, GNCN-t1-Sigma) demonstrate how the basic framework can be extended and improved, with GNCN-t1-Sigma showing particularly strong performance through its incorporation of variance parameters.

As research in this area continues, we can expect further advances in both the theoretical understanding and practical applications of Neural Generative Coding.
