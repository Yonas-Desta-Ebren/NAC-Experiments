# Mathematical Foundations of Generative Models

This document provides a detailed mathematical explanation of the generative models compared in our experiments: GNCN-PDH, GVAE, GVAE-CV, and RAE. Understanding these mathematical foundations is crucial for explaining how these models work during interviews.

## Table of Contents
1. [Variational Autoencoders (VAEs)](#1-variational-autoencoders-vaes)
2. [Gaussian Variational Autoencoder (GVAE)](#2-gaussian-variational-autoencoder-gvae)
3. [Gaussian VAE with Constant Variance (GVAE-CV)](#3-gaussian-vae-with-constant-variance-gvae-cv)
4. [Regularized Autoencoder (RAE)](#4-regularized-autoencoder-rae)
5. [Neural Generative Coding (NGC) and GNCN-PDH](#5-neural-generative-coding-ngc-and-gncn-pdh)
6. [Comparison of Learning Rules](#6-comparison-of-learning-rules)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Interview Preparation](#8-interview-preparation)

## 1. Variational Autoencoders (VAEs)

### 1.1 Probabilistic Framework

Variational Autoencoders (VAEs) are latent variable models that aim to learn the underlying data distribution $p(x)$ by introducing a latent variable $z$. The joint distribution is factorized as:

$$p(x, z) = p(x|z)p(z)$$

Where:
- $p(z)$ is the prior distribution over latent variables (typically a standard normal distribution)
- $p(x|z)$ is the likelihood of the data given the latent variables (decoder)

The marginal likelihood $p(x)$ is obtained by integrating over all possible values of $z$:

$$p(x) = \int p(x|z)p(z) dz$$

This integral is generally intractable, so VAEs use variational inference to approximate it.

### 1.2 Variational Inference

VAEs introduce an approximate posterior distribution $q_\phi(z|x)$ (encoder) to approximate the true posterior $p(z|x)$. The goal is to minimize the Kullback-Leibler (KL) divergence between these distributions:

$$\text{KL}(q_\phi(z|x) || p(z|x))$$

This can be reformulated to maximize the Evidence Lower Bound (ELBO):

$$\text{ELBO}(\phi, \theta) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) || p(z))$$

Where:
- $\phi$ are the parameters of the encoder network
- $\theta$ are the parameters of the decoder network
- The first term is the reconstruction term (how well the model reconstructs the input)
- The second term is the regularization term (how close the approximate posterior is to the prior)

### 1.3 Reparameterization Trick

To enable backpropagation through the sampling process, VAEs use the reparameterization trick. Instead of directly sampling from $q_\phi(z|x)$, we parameterize it as:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$$

Where:
- $\mu_\phi(x)$ is the mean of the approximate posterior
- $\sigma_\phi(x)$ is the standard deviation of the approximate posterior
- $\epsilon \sim \mathcal{N}(0, I)$ is a random noise vector
- $\odot$ denotes element-wise multiplication

This allows gradients to flow through the deterministic parameters $\mu_\phi(x)$ and $\sigma_\phi(x)$.

## 2. Gaussian Variational Autoencoder (GVAE)

### 2.1 Model Definition

In a Gaussian VAE, both the prior and approximate posterior are Gaussian distributions:

$$p(z) = \mathcal{N}(z; 0, I)$$
$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$$

The decoder defines a likelihood function, which for MNIST (binary data) is typically a Bernoulli distribution:

$$p_\theta(x|z) = \prod_{i=1}^D \text{Bern}(x_i; f_\theta(z)_i)$$

Where $f_\theta(z)$ is the output of the decoder network, representing the parameters of the Bernoulli distribution (probabilities).

### 2.2 Loss Function

The ELBO for the Gaussian VAE can be written as:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) || p(z))$$

For binary data like MNIST, the reconstruction term becomes the binary cross-entropy:

$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] \approx -\text{BCE}(x, \hat{x})$$

Where $\hat{x} = f_\theta(z)$ is the reconstructed input.

The KL divergence between two Gaussian distributions has a closed-form solution:

$$\text{KL}(q_\phi(z|x) || p(z)) = \frac{1}{2} \sum_{j=1}^J \left( \mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1 \right)$$

Where $J$ is the dimensionality of the latent space.

### 2.3 Network Architecture

The GVAE architecture consists of:

1. **Encoder Network**: Maps input $x$ to parameters of $q_\phi(z|x)$
   - Input: $x \in \mathbb{R}^D$ (e.g., 784-dimensional for MNIST)
   - Hidden layers: Fully connected layers with non-linear activations
   - Output: $\mu_\phi(x) \in \mathbb{R}^J$ and $\log \sigma^2_\phi(x) \in \mathbb{R}^J$

2. **Sampling Layer**: Implements the reparameterization trick
   - Input: $\mu_\phi(x)$ and $\sigma_\phi(x)$
   - Output: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$

3. **Decoder Network**: Maps latent variable $z$ to reconstructed input $\hat{x}$
   - Input: $z \in \mathbb{R}^J$
   - Hidden layers: Fully connected layers with non-linear activations
   - Output: $\hat{x} \in [0, 1]^D$ (probabilities for Bernoulli distribution)

## 3. Gaussian VAE with Constant Variance (GVAE-CV)

### 3.1 Model Modification

The GVAE-CV is a simplified version of the GVAE where the variance of the approximate posterior is fixed to a constant value:

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma^2 I)$$

Where $\sigma^2$ is a fixed hyperparameter (e.g., $\sigma^2 = 1$).

### 3.2 Loss Function Simplification

With a constant variance, the KL divergence term simplifies to:

$$\text{KL}(q_\phi(z|x) || p(z)) = \frac{1}{2} \sum_{j=1}^J \left( \mu_j^2 + \sigma^2 - \log(\sigma^2) - 1 \right)$$

If $\sigma^2 = 1$, this further simplifies to:

$$\text{KL}(q_\phi(z|x) || p(z)) = \frac{1}{2} \sum_{j=1}^J \mu_j^2$$

This is equivalent to L2 regularization on the means of the approximate posterior.

### 3.3 Network Architecture Simplification

The encoder network no longer needs to output the variance parameters, simplifying the architecture:

1. **Encoder Network**: Maps input $x$ to mean parameters of $q_\phi(z|x)$
   - Input: $x \in \mathbb{R}^D$
   - Hidden layers: Fully connected layers with non-linear activations
   - Output: $\mu_\phi(x) \in \mathbb{R}^J$

2. **Sampling Layer**: Uses fixed variance in the reparameterization trick
   - Input: $\mu_\phi(x)$
   - Output: $z = \mu_\phi(x) + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$ and $\sigma$ is fixed

3. **Decoder Network**: Same as in GVAE

## 4. Regularized Autoencoder (RAE)

### 4.1 Deterministic Framework

Unlike VAEs, Regularized Autoencoders (RAEs) are deterministic models that do not use a probabilistic framework. They directly map inputs to a latent representation and back:

$$z = f_\phi(x)$$
$$\hat{x} = g_\theta(z)$$

Where:
- $f_\phi$ is the encoder function with parameters $\phi$
- $g_\theta$ is the decoder function with parameters $\theta$

### 4.2 Loss Function

The loss function for RAE consists of a reconstruction term and a regularization term:

$$\mathcal{L}(\theta, \phi; x) = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \lambda \mathcal{L}_{\text{reg}}(x, z, \phi, \theta)$$

For binary data like MNIST, the reconstruction term is typically the binary cross-entropy:

$$\mathcal{L}_{\text{recon}}(x, \hat{x}) = \text{BCE}(x, \hat{x})$$

The regularization term can take various forms. In our implementation, we use L2 regularization on the weights:

$$\mathcal{L}_{\text{reg}}(x, z, \phi, \theta) = \|\phi\|_2^2 + \|\theta\|_2^2$$

Where $\|\cdot\|_2^2$ denotes the squared L2 norm.

### 4.3 Network Architecture

The RAE architecture consists of:

1. **Encoder Network**: Maps input $x$ to latent representation $z$
   - Input: $x \in \mathbb{R}^D$
   - Hidden layers: Fully connected layers with non-linear activations
   - Output: $z = f_\phi(x) \in \mathbb{R}^J$

2. **Decoder Network**: Maps latent representation $z$ to reconstructed input $\hat{x}$
   - Input: $z \in \mathbb{R}^J$
   - Hidden layers: Fully connected layers with non-linear activations
   - Output: $\hat{x} = g_\theta(z) \in [0, 1]^D$

## 5. Neural Generative Coding (NGC) and GNCN-PDH

### 5.1 Predictive Coding Framework

Neural Generative Coding (NGC) is based on the predictive coding theory from neuroscience. It models the brain as a hierarchical system that tries to predict its inputs and updates its internal states based on prediction errors.

In NGC, the network consists of multiple layers, each trying to predict the activity of the layer below it. The prediction errors drive the learning process.

### 5.2 GNCN-PDH Model

GNCN-PDH (Generative Neural Coding Network with Predictive Discrete Hebbian learning) is an implementation of NGC that uses local Hebbian-like learning rules.

The model consists of a hierarchy of layers, where each layer $l$ has:
- State variables $\mathbf{s}^l$
- Prediction of the layer below $\hat{\mathbf{s}}^{l-1} = f^l(\mathbf{s}^l)$
- Prediction error $\mathbf{e}^{l-1} = \mathbf{s}^{l-1} - \hat{\mathbf{s}}^{l-1}$

### 5.3 Learning Dynamics

The state update rule for layer $l$ is:

$$\Delta \mathbf{s}^l = \eta_s \left( \mathbf{W}^l \mathbf{e}^{l-1} - \mathbf{e}^l \right)$$

Where:
- $\eta_s$ is the state update learning rate
- $\mathbf{W}^l$ is the weight matrix for layer $l$
- $\mathbf{e}^{l-1}$ is the prediction error from the layer below
- $\mathbf{e}^l$ is the prediction error from the layer above

The weight update rule follows a Hebbian-like learning rule:

$$\Delta \mathbf{W}^l = \eta_W \mathbf{e}^{l-1} (\mathbf{s}^l)^T$$

Where $\eta_W$ is the weight update learning rate.

This learning rule is local, meaning that the update for a weight depends only on the activities of the neurons it connects, making it more biologically plausible than backpropagation.

### 5.4 Predictive Discrete Hebbian (PDH) Learning

PDH is a specific implementation of the Hebbian learning rule that includes:

1. **Discrete Updates**: States are updated in discrete steps rather than continuously
2. **Predictive Component**: Updates are based on prediction errors
3. **Hebbian Learning**: Weight updates depend on the correlation between pre- and post-synaptic activities

The PDH learning rule can be written as:

$$\Delta \mathbf{W}^l = \eta_W \mathbf{e}^{l-1} (\mathbf{s}^l)^T - \lambda \mathbf{W}^l$$

Where $\lambda$ is a weight decay parameter for regularization.

## 6. Comparison of Learning Rules

### 6.1 Backpropagation (Used in GVAE, GVAE-CV, RAE)

Backpropagation computes gradients of the loss function with respect to all parameters by applying the chain rule of calculus:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial \hat{x}} \frac{\partial \hat{x}}{\partial \theta}$$
$$\frac{\partial \mathcal{L}}{\partial \phi} = \frac{\partial \mathcal{L}}{\partial z} \frac{\partial z}{\partial \phi}$$

This requires propagating error signals backward through the network, which is not considered biologically plausible because:
1. It requires symmetric weights between forward and backward passes
2. It needs to store activations from the forward pass
3. It involves non-local computations

### 6.2 Hebbian Learning (Used in GNCN-PDH)

Hebbian learning follows the principle "neurons that fire together, wire together." The basic Hebbian rule is:

$$\Delta W_{ij} = \eta x_i y_j$$

Where:
- $W_{ij}$ is the weight connecting neurons $i$ and $j$
- $x_i$ is the activity of the pre-synaptic neuron
- $y_j$ is the activity of the post-synaptic neuron
- $\eta$ is the learning rate

This is a local learning rule, meaning the update for a weight depends only on the activities of the neurons it connects, making it more biologically plausible.

### 6.3 Predictive Coding (Used in GNCN-PDH)

Predictive coding combines Hebbian learning with prediction errors. The weight update rule is:

$$\Delta W_{ij} = \eta e_j x_i$$

Where $e_j$ is the prediction error at neuron $j$.

This learning rule is still local and more biologically plausible than backpropagation, while potentially being more powerful than simple Hebbian learning.

## 7. Evaluation Metrics

### 7.1 Binary Cross-Entropy (BCE)

BCE measures the reconstruction quality of the models. For binary data like MNIST, it is defined as:

$$\text{BCE}(x, \hat{x}) = -\frac{1}{D} \sum_{i=1}^D [x_i \log(\hat{x}_i) + (1 - x_i) \log(1 - \hat{x}_i)]$$

Where:
- $x$ is the original input
- $\hat{x}$ is the reconstructed input
- $D$ is the dimensionality of the input

Lower BCE values indicate better reconstruction performance.

### 7.2 Masked Mean Squared Error (M-MSE)

M-MSE measures the model's ability to reconstruct partially masked inputs. It is defined as:

$$\text{M-MSE}(x, \hat{x}_m) = \frac{1}{|M|} \sum_{i \in M} (x_i - \hat{x}_{m,i})^2$$

Where:
- $M$ is the set of masked pixels
- $\hat{x}_m$ is the reconstruction of the masked input
- $|M|$ is the number of masked pixels

Lower M-MSE values indicate better generalization to incomplete data.

### 7.3 Classification Error

Classification error measures the model's ability to learn discriminative features in the latent space. It is typically computed by training a classifier on the latent representations and evaluating its performance:

$$\text{Classification Error} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(y_i \neq \hat{y}_i)$$

Where:
- $y_i$ is the true label
- $\hat{y}_i$ is the predicted label
- $\mathbf{1}(\cdot)$ is the indicator function
- $N$ is the number of test samples

Lower classification error indicates better representation learning.

### 7.4 Log-Likelihood

Log-likelihood measures the model's ability to capture the underlying data distribution. For VAEs, it is estimated using importance sampling:

$$\log p(x) \approx \log \left( \frac{1}{K} \sum_{k=1}^K \frac{p(x|z_k)p(z_k)}{q(z_k|x)} \right)$$

Where:
- $z_k \sim q(z|x)$ are samples from the approximate posterior
- $K$ is the number of samples

Higher (less negative) log-likelihood values indicate better density estimation.

## 8. Interview Preparation

### 8.1 Key Concepts to Understand

1. **Variational Inference**: Understand how VAEs approximate the intractable posterior using variational inference
2. **Reparameterization Trick**: Be able to explain how this enables backpropagation through random sampling
3. **ELBO**: Understand the components of the Evidence Lower Bound and what they represent
4. **Regularization**: Explain the different regularization approaches in VAEs vs. RAEs
5. **Biological Plausibility**: Understand the limitations of backpropagation and how Hebbian learning addresses them
6. **Predictive Coding**: Be familiar with the predictive coding theory and how it relates to NGC
7. **Evaluation Metrics**: Know what each metric measures and how to interpret the results

### 8.2 Common Interview Questions

1. **What is the difference between a VAE and a standard autoencoder?**
   - VAEs are probabilistic models that learn a distribution over latent variables, while standard autoencoders learn a deterministic mapping.
   - VAEs include a regularization term (KL divergence) that encourages the latent space to follow a prior distribution.
   - VAEs can generate new samples by sampling from the prior, while standard autoencoders cannot.

2. **Explain the reparameterization trick in VAEs.**
   - The reparameterization trick allows gradients to flow through the sampling process.
   - Instead of directly sampling from the approximate posterior, we parameterize it as a deterministic function of the input and a random noise variable.
   - This enables backpropagation through the sampling process, which is necessary for training VAEs.

3. **What is the role of the KL divergence term in the VAE objective?**
   - The KL divergence term regularizes the approximate posterior to be close to the prior.
   - It prevents the model from simply memorizing the training data by encouraging the latent representations to follow a specific distribution.
   - It also enables generation of new samples by sampling from the prior.

4. **How does GVAE-CV differ from GVAE, and what are the implications?**
   - GVAE-CV uses a fixed variance for the approximate posterior, while GVAE learns it.
   - This simplifies the model and reduces the number of parameters.
   - It can lead to better reconstruction performance but potentially worse density estimation.

5. **What makes Hebbian learning more biologically plausible than backpropagation?**
   - Hebbian learning is local, meaning the update for a weight depends only on the activities of the neurons it connects.
   - It doesn't require storing activations from the forward pass.
   - It doesn't need symmetric weights between forward and backward passes.

6. **Explain the predictive coding theory and how it relates to NGC.**
   - Predictive coding posits that the brain tries to predict its inputs and updates its internal states based on prediction errors.
   - NGC implements this by having each layer predict the activity of the layer below it.
   - Learning is driven by minimizing prediction errors using local Hebbian-like rules.

7. **Why might RAE perform better on reconstruction but worse on density estimation compared to VAEs?**
   - RAEs directly optimize for reconstruction without the constraints of a probabilistic framework.
   - They don't have the regularization from the KL divergence term, which can lead to better reconstruction but worse generalization.
   - VAEs make trade-offs between reconstruction and regularization, which can lead to better density estimation but worse reconstruction.

8. **How would you choose between these models for a specific application?**
   - If biological plausibility is important, GNCN-PDH would be preferred.
   - If pure reconstruction quality is the goal, RAE performs best.
   - If probabilistic modeling and density estimation are important, GVAE is a better choice.
   - GVAE-CV offers a good middle ground between reconstruction quality and probabilistic modeling.

### 8.3 Mathematical Derivations to Practice

1. **Derive the ELBO for VAEs**
2. **Derive the KL divergence between two Gaussian distributions**
3. **Show how the reparameterization trick enables gradient computation**
4. **Derive the gradient updates for the GNCN-PDH model**
5. **Explain how to estimate the log-likelihood using importance sampling**

### 8.4 Implementation Details to Understand

1. **Network architectures for each model**
2. **Hyperparameter choices and their impact**
3. **Training procedures and optimization algorithms**
4. **Evaluation protocols for each metric**
5. **Computational efficiency considerations**

By thoroughly understanding these concepts, you will be well-prepared to discuss these models in an interview setting and demonstrate your expertise in generative modeling.
