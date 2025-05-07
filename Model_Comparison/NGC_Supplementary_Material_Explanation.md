# Detailed Explanation of "Neural Generative Coding Framework for Learning Generative Models" Supplementary Material

## 1. NGC Framework and Naming Convention

The document establishes a clear naming convention for Generative Neural Coding Network (GNCN) models:

- **Base Name**: GNCN (Generative Neural Coding Network)
- **Error Synapse Types**:
  - **Type 1 (-t1)**: Error synapses that are functions of forward generative weights (virtual synapses)
  - **Type 2 (-t2)**: Separate learnable synaptic parameters dedicated to transmitting error messages

- **Additional Suffixes**:
  - **-L**: Indicates lateral synapses in state variables
  - **-Σ**: Indicates lateral precision weights in error neurons
  - **-PDH**: "Partially decomposable hierarchy" - used when the model has a non-hierarchical structure in its generative/prediction neural structure (αm = 1)
  - **-PDEH**: "Partially decomposable error hierarchy" - used when the model contains a non-hierarchical error structure

## 2. Four Key NGC Models Investigated

The document details four main GNCN models that were investigated:

1. **GNCN-t1/Rao**: 
   - Type 1 error synapses
   - No precision weights
   - No lateral weights
   - αm = 0
   - Uses partial derivatives of activation functions
   - Recovers the classical predictive coding model of Rao & Ballard

2. **GNCN-t1-Σ/Friston**:
   - Type 1 error synapses
   - Has precision weights
   - No lateral weights
   - αm = 0
   - Uses partial derivatives of activation functions
   - Recovers Friston's predictive coding model

3. **GNCN-t2-LΣ**:
   - Type 2 error synapses (learnable)
   - Has precision weights
   - Has lateral weights
   - αm = 0
   - Does not use partial derivatives of activation functions
   - Novel model with enhanced biological plausibility

4. **GNCN-t2-LΣ-PDH** (abbreviated as **GNCN-PDH**):
   - Type 2 error synapses
   - Has precision weights
   - Has lateral weights
   - αm = 1 (partially decomposable hierarchy)
   - Does not use partial derivatives of activation functions
   - Most advanced model with non-hierarchical structure

## 3. Variant NGC Architectures

The document describes five variant NGC architectures:

1. **Multi-input GNCN-t2-Σ**: Handles multiple clamped inputs to generate/predict outputs (useful for direct classification tasks)

2. **Multimodal GNCN-t2-LΣ**: For multi-modal generative modeling (e.g., jointly learning to synthesize an image and discrete one-hot encoding of a word/character)

3. **GNCN-t2-LΣ-PDEH (GNCN-PDEH)**: A generative model where upper layers receive error messages from layers other than its immediately connected one

4. **GNCN-t2-LΣ-PDH (label aware)**: A label-aware generative model that forms a partially decomposable hierarchy in its forward generative structure

5. **GNCN-t2-Σ-RecN**: A temporal/recurrent NGC model where predictive outputs of each state region are conditioned on their previous values

## 4. Model Hyperparameters and Architecture Details

The document provides detailed information about model hyperparameters:

- **Latent Dimensions**: 
  - For NGC models with lateral synapses: 20 neural columns in topmost layers
  - Lower levels: 360 neurons (found optimal through experimentation)
  - GNCN-t1 and GNCN-t1-Σ: 360 neurons in top layer

- **Activation Functions**:
  - Linear rectifier (ReLU) for NGC models to ensure positive activity values
  - GNCN-t1-Σ: Linear rectifier worked best
  - GNCN-t1: Hyperbolic tangent worked best

- **Comparison with Autoencoders**:
  - Autoencoder hidden layer sizes were set to be equal
  - Maximum parameter count was constrained to match NGC models (approximately 1,400,000 synapses)

## 5. Computational Complexity and Run-time Considerations

The document addresses the computational complexity of NGC models:

- **Inference Cost**:
  - NGC models require multiple steps of processing (T iterations) for inference
  - Autoencoder: ~2L matrix multiplications (L = number of layers)
  - NGC models: ~2L×T matrix multiplications
  - This makes NGC models slower per sample than feedforward autoencoders

- **Efficiency Considerations**:
  - NGC models converge with fewer samples than backprop models
  - Specialized hardware could exploit NGC's inherent parallelism
  - Potential solutions include designing alternative state update equations or amortized inference processes

## 6. Omission of Activation Derivatives

The document explains why activation derivatives were omitted in GNCN-t2-LΣ and GNCN-PDH models:

- **Stability Considerations**:
  - No strong weight fluctuations were observed in simulations
  - Weight columns are constrained to have unit norms
  - Step size β is kept within [0.02, 0.1]
  - Leak variable -γzℓ helps smooth values and prevent large magnitudes

- **Theoretical Justification**:
  - As long as the activation function is monotonically increasing, the learning process remains stable
  - The benefit of the point-wise derivative is absorbed by the error synaptic weights

## 7. Lateral Competition Matrices

The document details how lateral competition matrices are generated:

- **Matrix Equation**: Vℓ = αh(Mℓ) ⊙ (1-I) - αe(I)
  - I is the identity matrix
  - Mℓ is a masking matrix set by the experimenter
  - αe = 0.13 (self-excitation strength)
  - αh = 0.125 (lateral inhibition strength)

- **Mask Matrix Generation Process**:
  1. Create Jℓ/K matrices of shape Jℓ×K of zeros
  2. Insert ones at specific coordinates to create the desired inhibition pattern
  3. Concatenate the matrices along the horizontal axis

- **Biological Plausibility**:
  - While not directly justified in the probabilistic model, experiments show lateral synapses improve performance
  - Future work will derive a probabilistic interpretation of these extensions

## 8. Autoencoder Baseline Model Descriptions

The document provides detailed descriptions of the autoencoder baseline models used for comparison:

1. **Regularized Auto-encoder (RAE)**:
   - Standard autoencoder with L2 regularization
   - Encoder maps input x to latent representation z
   - Decoder reconstructs input from z
   - Uses linear rectifier activations with logistic sigmoid at output layer

2. **Gaussian Variational Auto-encoder (GVAE)**:
   - Encoder produces parameters of a Gaussian distribution over z
   - Includes KL divergence term to match prior distribution
   - Uses reparameterization trick for sampling
   - Objective includes reconstruction term and KL divergence term

3. **Constant-Variance Gaussian Variational Auto-encoder (GVAE-CV)**:
   - Similar to GVAE but with fixed variance parameter
   - Variance meta-parameter chosen from [0.025, 1.0] based on validation performance

4. **Generative Adversarial Network Autoencoder (GAN-AE)**:
   - Also called adversarial autoencoder
   - Similar to GVAE but replaces KL divergence with adversarial objective
   - Includes discriminator network to distinguish between prior samples and encoder outputs
   - Uses multi-step optimization process

## 9. Feature Analysis of Neural Generative Coding

The document describes a feature analysis conducted on the GNCN-t2-LΣ model:

- **Layer 1 to Layer 0 Features**:
  - Resembled rough strokes and digit components of different orientations/translations

- **Higher Layer Features**:
  - Weight vectors in layers 2 and 3 resembled neural selection "blueprints" or maps
  - These maps appear to select or trigger lower-level state neurons

- **Multi-level Command Structure**:
  - NGC models learn a hierarchical command structure
  - Neurons in higher levels turn on/off neurons in lower levels
  - Intensity coefficients scale the activation of selected neurons
  - The final composition of low-level features produces complete objects/digits

- **Comparison to Sparse Coding**:
  - NGC learns to compose and produce a weighted summation of low-level features
  - Similar to sparse coding but driven by a complex, higher-level neural latent structure

## 10. Experimental Results and Visualizations

The document includes several supplementary figures showing:

- **Class Distribution Visualization**: Approximate label distributions produced by each model on MNIST, KMNIST, FMNIST, and CalTech datasets
- **Nearest Neighbor Analysis**: Comparing samples generated by NGC models with backpropagation-based autoencoder models
- **Feature Visualization**: Illustrating how higher-level maps interact with low-level visual features

## 11. Key Advantages of NGC Models

From the document, several advantages of NGC models can be identified:

1. **Biological Plausibility**:
   - More aligned with neural processing in the brain
   - Local learning rules instead of backpropagation
   - No need for activation function derivatives in advanced models

2. **Data Efficiency**:
   - Converge with fewer samples than backprop models
   - Better generalization from limited data

3. **Hierarchical Representation**:
   - Learn meaningful hierarchical features
   - Higher layers control and organize lower-level features
   - Create compositional representations

4. **Flexibility**:
   - Can be extended to various architectures (multi-input, multimodal, recurrent)
   - Compatible with different distribution assumptions
   - Adaptable to different tasks beyond reconstruction
