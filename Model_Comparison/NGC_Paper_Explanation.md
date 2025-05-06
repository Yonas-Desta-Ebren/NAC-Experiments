# Neural Generative Coding (NGC): Deep Dive Explanation

## 1. Introduction and Core Concepts

Neural Generative Coding (NGC) is a computational framework for developing neural generative models inspired by the theory of predictive processing in the brain. This document provides a detailed explanation of the key concepts, mathematical formulations, and practical implementations of NGC models based on the paper "Neural Generative Coding Through Probabilistic Feedback" (2012.03405v4).

### 1.1 Predictive Processing in the Brain

The fundamental idea behind NGC comes from the theory of predictive processing in neuroscience, which posits that:

- The brain is constantly trying to predict its sensory inputs
- Neurons in the brain form a hierarchy where neurons in one level form expectations about sensory inputs from another level
- These neurons update their local models based on differences between their expectations and the observed signals
- The brain is essentially a hierarchical generative model that is continuously making predictions and self-correcting

### 1.2 Limitations of Traditional Backpropagation

Traditional neural networks trained with backpropagation have several limitations from a biological plausibility perspective:

1. **Weight Transport Problem**: Synapses that make up the forward information pathway need to directly be used in reverse to communicate teaching signals
2. **Derivative Problem**: Neurons need to be able to know and communicate their own activation function's first derivative
3. **Update-Locking Problem**: Neurons must wait for neurons ahead of them to percolate their error signals back
4. **Global Feedback Pathway Problem**: There is a distinct form of information propagation for error feedback that doesn't influence neural activity
5. **One-to-One Correspondence**: Error signals have a one-to-one correspondence with neurons

NGC addresses the first four of these limitations by proposing a more biologically plausible learning framework.

## 2. NGC Framework and Mathematical Formulation

### 2.1 Model Architecture

A Generative Neural Coding Network (GNCN) has L+1 layers of neurons (state variables) N₀, N₁, ..., Nₗ, where:

- N₀ is the output layer (typically clamped to the input data x)
- Each layer Nₗ has Jₗ neurons, each with a latent state value
- The combined latent state of all neurons in layer Nₗ is represented by the vector zₗ ∈ ℝᴶₗ×¹
- The network specifies a probability P(z₀=x, z₁, ..., zₗ) = P(z₀|z₁)...P(zₗ₋₁|zₗ)P(zₗ)

### 2.2 Generative Process

The generative process in NGC is modeled as:

zₗ ≈ gₗ(Wₗ₊₁ · Φₗ₊₁(zₗ₊₁) + αₘ · Mₗ₊₂ · Φₗ₊₂(zₗ₊₂))

Where:
- Wₗ is a forward/generative weight matrix
- αₘ is a binary coefficient (0 or 1)
- gₗ and Φₗ₊₁ are activation functions
- Mₗ₊₂ is an optional auxiliary generative matrix

When αₘ = 1, the model is called "PDH" (partially decomposable hierarchy).

### 2.3 Error Neurons and Prediction Errors

A key innovation in NGC is the introduction of error neurons eₗ that compute prediction errors:

e₀ = (x ⊘ z₀ - (1-x) ⊘ (1-z₀))

eₗ = (Σₗ₋₁)⁻¹ ⊙ (zₗ₋₁ - z̄ₗ₋₁)

Where:
- ⊘ is element-wise division
- ⊙ is element-wise product
- Σₗ is a covariance matrix that acts as lateral modulation

### 2.4 State Update Rule

The state neurons are updated based on a combination of:
1. A leak term (decay pressure)
2. Top-down pressure from error neurons
3. Bottom-up pressure from error neurons in the layer below
4. Lateral inhibition/excitation

The update rule is:

zₗᵢ ← zₗᵢ + β(-γzₗᵢ + (∑ⱼ∈ₗ₋₁ Eₗᵢⱼeₗ₋₁ⱼ) - eₗᵢ - (∑ⱼ∈ₗ,ⱼ≠ᵢ Vₗᵢⱼ·Φₗ(zₗⱼ)) + Vₗᵢᵢ·Φₗ(zₗᵢ))

Where:
- γ controls the strength of the decay/leak
- β is a learning rate parameter
- Eₗ is a matrix of error synapses
- Vₗ is a matrix of lateral excitatory/inhibitory synapses

### 2.5 Synaptic Update Rules

The synaptic weights are updated using Hebbian-like learning rules:

∂W₀/∂t ∝ e₀ · (Φ₁(z₁))ᵀ
∂Wₗ/∂t ∝ eₗ · (Φₗ₊₁(zₗ₊₁))ᵀ
∂Eₗ/∂t ∝ η(Φₗ₊₁(zₗ₊₁) · (eₗ)ᵀ)

These update rules are local and don't require the computation of activation function derivatives, making them more biologically plausible.

## 3. NGC Model Variants

The NGC framework encompasses several model variants:

### 3.1 GNCN-t1/Rao

This model recovers the classical predictive coding model proposed by Rao & Ballard (1999) with:
- αₘ = 0
- E = (W)ᵀ (error synapses are transpose of generative weights)
- Σₗ = σ²I (covariance is a scalar times identity matrix)
- Φₗ(v) = tanh(v) (hyperbolic tangent activation)

### 3.2 GNCN-t1-Σ/Friston

This model recovers the neural implementation of the graphical model proposed by Friston (2008) with:
- αₘ = 0
- E = (W)ᵀ
- Φₗ(v) = max(0,v) (ReLU activation)

### 3.3 GNCN-t2-LΣ

This model uses:
- Separate, learnable error synapses E
- Lateral connectivity matrices V for competition among neurons
- Precision-weighted error computation

### 3.4 GNCN-PDH

This model extends GNCN-t2-LΣ by:
- Setting αₘ = 1 to enable auxiliary generative matrices M
- Creating a partially decomposable hierarchy

## 4. Training Process

The training process for NGC models follows an online alternating maximization approach:

1. **Initialization**:
   - Set z₀ = x (clamp data to output neurons)
   - Set zₗ = 0 for all other layers
   - Compute initial predictions and error neurons

2. **Latent Update Step** (E-step analog):
   - Update state neurons based on error signals
   - Update error neurons based on new states
   - Repeat for T iterations

3. **Parameter Update Step** (M-step analog):
   - Update synaptic matrices using Hebbian-like rules
   - Normalize weight matrices

This process embodies the idea that neural state and synaptic weight adjustment are the result of a process of generate-then-correct, or continual error correction, in response to samples of the environment.

## 5. Metaphorical Examples

### 5.1 The Orchestra Metaphor

Imagine a symphony orchestra where:

- **Conductor (Top Layer)**: Has a score (internal model) and directs the musicians
- **Musicians (Lower Layers)**: Try to play according to the conductor's directions
- **Sound Produced (Output)**: The actual music that is played
- **Listening (Error Computation)**: Musicians listen to what they and others are playing
- **Adjustments (Error Correction)**: Musicians adjust their playing based on what they hear

In this metaphor:
- The conductor's score represents the top-level internal model
- The musicians represent neurons in different layers
- The sound produced represents the activity of each layer
- The discrepancies between expected and actual sounds represent prediction errors
- The rehearsal process represents the training process

### 5.2 The Weather Forecasting Metaphor

Consider a team of weather forecasters organized hierarchically:

- **Senior Forecaster (Top Layer)**: Makes high-level predictions about weather patterns
- **Junior Forecasters (Middle Layers)**: Make more detailed predictions based on the senior forecaster's guidance
- **Field Observers (Bottom Layer)**: Collect actual weather data
- **Prediction Errors**: Differences between forecasts and actual weather
- **Model Updates**: Adjustments to forecasting models based on prediction errors

In this metaphor:
- The forecasting models represent the internal models at each layer
- The hierarchical organization represents the layered structure of the network
- The prediction errors drive learning and model updating
- The forecasters' models become more accurate over time, just as the neural network learns

### 5.3 The Building Construction Metaphor

Think of constructing a building:

- **Architect (Top Layer)**: Has the overall design and vision
- **Project Managers (Middle Layers)**: Translate the design into specific plans
- **Construction Workers (Bottom Layer)**: Execute the plans
- **Inspections (Error Computation)**: Compare the actual construction to the plans
- **Corrections (Error Feedback)**: Make adjustments when construction deviates from plans

In this metaphor:
- The architectural plans represent the top-level model
- The hierarchy of workers represents the layers of neurons
- The actual building represents the output
- The inspections represent error computation
- The corrections represent the error-driven learning process

## 6. Advantages of NGC Over Traditional Models

### 6.1 Biological Plausibility

NGC models are more biologically plausible because:

1. They use local learning rules instead of backpropagation
2. They don't require neurons to know their own activation derivatives
3. They don't suffer from the weight transport problem
4. They integrate error signals directly into neural activity
5. They incorporate lateral connectivity for contextual processing

### 6.2 Performance Benefits

According to the paper, NGC models:

1. Remain competitive with or outperform backprop-based models like VAEs on generative tasks
2. Perform well on tasks they weren't directly trained for, such as classification and pattern completion
3. Benefit from improvements like learnable recurrent error synapses and laterally-driven sparsity

### 6.3 Theoretical Connections

NGC connects to several important theoretical frameworks:

1. **Free Energy Principle**: NGC can be viewed as minimizing variational free energy
2. **Bayesian Brain Hypothesis**: NGC performs a form of approximate Bayesian inference
3. **Predictive Processing Theory**: NGC implements the core ideas of predictive processing

## 7. Practical Implementation Considerations

When implementing NGC models, several practical considerations are important:

### 7.1 Initialization

- Initialize weights with small random values
- Initialize state variables to zero
- Clamp the output layer to the input data

### 7.2 Hyperparameters

Key hyperparameters include:
- β: Controls the state update rate
- γ: Controls the strength of the leak term
- T: Number of iterations for the latent update step
- η: Learning rate for error synapses

### 7.3 Lateral Connectivity

The lateral connectivity matrix V can be designed to create different competition patterns:
- Self-excitation (diagonal elements)
- Lateral inhibition (off-diagonal elements)
- Group competition (block structure)

### 7.4 Sampling from the Model

To generate samples from a trained NGC model:
1. Initialize the top layer zₗ with random noise
2. Run the iterative settling process for T steps
3. Read out the values from the output layer z₀

## 8. Conclusion

Neural Generative Coding provides a powerful framework for developing generative models that are more biologically plausible than traditional backpropagation-based approaches. By incorporating principles from predictive processing theory, NGC models learn through local error correction and lateral competition, resulting in effective generative models that can compete with or outperform traditional approaches.

The different variants of NGC models (GNCN-t1/Rao, GNCN-t1-Σ/Friston, GNCN-t2-LΣ, GNCN-PDH) demonstrate the flexibility of the framework and its ability to unify different approaches to predictive coding. The incorporation of learnable error synapses and lateral connectivity further enhances the capabilities of these models.

As research in this area continues, NGC offers promising avenues for developing more brain-like artificial intelligence systems that can learn efficiently from their environment without the need for explicit supervision.
