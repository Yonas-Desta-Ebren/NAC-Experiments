# Mathematical Explanations of Neural Generative Coding (NGC) Concepts

## 1. Generative Neural Coding Network (GNCN) Mathematical Framework

### 1.1 Probabilistic Generative Model

The NGC framework is based on a hierarchical probabilistic generative model. For a network with L+1 layers of neurons (state variables) N₀, N₁, ..., Nₗ:

- The joint probability distribution is factorized as:
  
  $P(z_0=x, z_1, ..., z_L) = P(z_0|z_1)P(z_1|z_2)...P(z_{L-1}|z_L)P(z_L)$

- Each conditional probability is typically modeled as a Gaussian distribution:
  
  $P(z_{\ell-1}|z_\ell) = \mathcal{N}(z_{\ell-1}; g_\ell(W_\ell \cdot \Phi_\ell(z_\ell)), \Sigma_{\ell-1})$

Where:
- $z_\ell \in \mathbb{R}^{J_\ell \times 1}$ is the state vector at layer $\ell$
- $g_\ell$ is an activation function
- $\Phi_\ell$ is a nonlinear transformation
- $W_\ell$ is the generative weight matrix
- $\Sigma_{\ell-1}$ is the covariance matrix

### 1.2 Generative Process

The generative process in standard NGC models is:

$\hat{z}_{\ell-1} = g_\ell(W_\ell \cdot \Phi_\ell(z_\ell))$

For the PDH variant (partially decomposable hierarchy), this becomes:

$\hat{z}_{\ell-1} = g_\ell(W_\ell \cdot \Phi_\ell(z_\ell) + \alpha_m \cdot M_{\ell+1} \cdot \Phi_{\ell+1}(z_{\ell+1}))$

Where:
- $\hat{z}_{\ell-1}$ is the prediction of layer $\ell-1$ from layer $\ell$
- $\alpha_m$ is a binary coefficient (0 or 1)
- $M_{\ell+1}$ is an auxiliary generative matrix

## 2. Error Computation

### 2.1 Error Neurons

Error neurons compute the prediction errors between actual and predicted states:

For Gaussian distributions (GNCN-t1/Rao and GNCN-t1-Σ/Friston):
$e_\ell = (\Sigma_\ell)^{-1} \odot (z_{\ell-1} - \hat{z}_{\ell-1})$

For the output layer with Bernoulli distribution:
$e_0 = (x \oslash \hat{z}_0 - (1-x) \oslash (1-\hat{z}_0))$

Where:
- $\odot$ is element-wise multiplication
- $\oslash$ is element-wise division
- $\Sigma_\ell$ is the covariance matrix (precision-weighting)

### 2.2 Precision Matrices

In models with precision weighting (GNCN-t1-Σ/Friston, GNCN-t2-LΣ, GNCN-PDH):

$\Sigma_\ell = \text{diag}(\sigma_\ell^2)$

Where $\sigma_\ell^2$ are learnable precision parameters that modulate the influence of prediction errors.

## 3. State Update Rules

### 3.1 General State Update Equation

The state update rule for neurons in layer $\ell$ is:

$z_\ell^{t+1} = z_\ell^t + \beta \Delta z_\ell^t$

Where:
- $\beta$ is the learning rate
- $\Delta z_\ell^t$ is the update direction

### 3.2 Update Direction Calculation

For GNCN-t1/Rao and GNCN-t1-Σ/Friston:

$\Delta z_\ell^t = -\gamma z_\ell^t + (W_\ell^T \cdot e_{\ell-1}^t) \odot \Phi_\ell'(z_\ell^t) - e_\ell^t$

For GNCN-t2-LΣ and GNCN-PDH (without activation derivatives):

$\Delta z_\ell^t = -\gamma z_\ell^t + (E_\ell \cdot e_{\ell-1}^t) - e_\ell^t - V_\ell \cdot \Phi_\ell(z_\ell^t)$

Where:
- $\gamma$ is the leak/decay parameter
- $W_\ell^T$ is the transpose of the generative weights
- $E_\ell$ is the learnable error synaptic matrix
- $\Phi_\ell'(z_\ell^t)$ is the derivative of the activation function
- $V_\ell$ is the lateral connectivity matrix

## 4. Lateral Competition

### 4.1 Lateral Connectivity Matrix

The lateral connectivity matrix $V_\ell$ is constructed as:

$V_\ell = \alpha_h(M_\ell) \odot (1-I) - \alpha_e(I)$

Where:
- $I$ is the identity matrix
- $M_\ell$ is a masking matrix
- $\alpha_e$ is the self-excitation strength (typically 0.13)
- $\alpha_h$ is the lateral inhibition strength (typically 0.125)

### 4.2 Mask Matrix Generation

The mask matrix $M_\ell$ for group competition is generated by:

1. Creating $J_\ell/K$ matrices of shape $J_\ell \times K$ of zeros: $\{S_1, S_2, ..., S_k, ..., S_C\}$ (where $C = J_\ell/K$)
2. In each matrix $S_k$, inserting ones at coordinates $c = \{1, ..., k, ..., K\}$ and $r = \{1 + K*(k-1), ..., k+K*(k-1), ..., K+K*(k-1)\}$
3. Concatenating the matrices horizontally: $M_\ell = \langle S_1, S_2, ..., S_C \rangle$

## 5. Synaptic Weight Updates

### 5.1 Generative Weight Updates

For GNCN-t1/Rao and GNCN-t1-Σ/Friston:

$\Delta W_\ell = \eta \cdot e_{\ell-1} \cdot (\Phi_\ell(z_\ell))^T$

For GNCN-t2-LΣ and GNCN-PDH:

$\Delta W_\ell = \eta \cdot e_{\ell-1} \cdot (\Phi_\ell(z_\ell))^T$

Where $\eta$ is the learning rate for weight updates.

### 5.2 Error Synaptic Updates (Type 2 models)

For GNCN-t2-LΣ and GNCN-PDH:

$\Delta E_\ell = \eta \cdot z_\ell \cdot (e_{\ell-1})^T$

### 5.3 Precision Parameter Updates

For models with precision weighting:

$\Delta \sigma_\ell^2 = \eta_\sigma \cdot ((e_\ell)^2 - \frac{1}{\sigma_\ell^2})$

Where $\eta_\sigma$ is the learning rate for precision parameters.

### 5.4 Weight Normalization

After each update, weight columns are normalized to have unit norm:

$W_\ell[:,j] = \frac{W_\ell[:,j]}{||W_\ell[:,j]||_2}$

## 6. Non-Gaussian Distributions in NGC

### 6.1 Student's t-Distribution

When using Student's t-distribution instead of Gaussian:

$P(z_{\ell-1}|z_\ell) = \text{St}(z_{\ell-1}; g_\ell(W_\ell \cdot \Phi_\ell(z_\ell)), \Sigma_{\ell-1}, \nu)$

Where $\nu$ is the degrees of freedom parameter.

The error computation becomes:

$e_\ell = \frac{(\nu + d_\ell)}{(\nu + \delta_\ell^2)} \cdot (\Sigma_\ell)^{-1} \odot (z_{\ell-1} - \hat{z}_{\ell-1})$

Where:
- $d_\ell$ is the dimensionality of layer $\ell$
- $\delta_\ell^2 = (z_{\ell-1} - \hat{z}_{\ell-1})^T \cdot (\Sigma_\ell)^{-1} \cdot (z_{\ell-1} - \hat{z}_{\ell-1})$

### 6.2 Laplace Distribution

When using Laplace distribution:

$P(z_{\ell-1}|z_\ell) = \text{Laplace}(z_{\ell-1}; g_\ell(W_\ell \cdot \Phi_\ell(z_\ell)), b_\ell)$

Where $b_\ell$ is the scale parameter.

The error computation becomes:

$e_\ell = \frac{1}{b_\ell} \cdot \text{sign}(z_{\ell-1} - \hat{z}_{\ell-1})$

### 6.3 Mixture of Gaussians

When using a mixture of Gaussians:

$P(z_{\ell-1}|z_\ell) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(z_{\ell-1}; \mu_k, \Sigma_k)$

Where:
- $\pi_k$ are the mixture weights
- $\mu_k = g_\ell(W_{\ell,k} \cdot \Phi_\ell(z_\ell))$
- $\Sigma_k$ are component-specific covariance matrices

The error computation involves responsibility-weighted errors from each component.

## 7. Inference Process

### 7.1 Iterative Settling

The inference process involves iteratively updating the state variables until convergence:

1. Initialize $z_0 = x$ (clamp input)
2. Initialize $z_\ell = 0$ for $\ell > 0$
3. For $t = 1$ to $T$:
   - Compute predictions $\hat{z}_{\ell-1}$ for all layers
   - Compute errors $e_\ell$ for all layers
   - Update states $z_\ell$ using the update rules
4. Return final states $z_\ell$

### 7.2 Sampling from the Model

To generate samples from a trained NGC model:

1. Initialize $z_L$ with random noise
2. Run the iterative settling process for $T$ steps
3. Read out the values from $z_0$

## 8. Comparison with Autoencoder Models

### 8.1 Regularized Autoencoder (RAE)

The objective function for RAE:

$\psi = \sum_j [x[j] \log z_0[j] + (1-x[j]) \log(1-z_0[j])] - \lambda \sum_{W_\ell \in \Theta} ||W_\ell||_F^2$

Where:
- $z_0$ is the output of the decoder
- $\lambda$ is the regularization strength
- $||W_\ell||_F$ is the Frobenius norm of weight matrix $W_\ell$

### 8.2 Gaussian Variational Autoencoder (GVAE)

The objective function for GVAE:

$\psi = \sum_j [x[j] \log z_0[j] + (1-x[j]) \log(1-z_0[j])] - D_{KL}[q(z|x) || p(z)]$

Where:
- $q(z|x) = \mathcal{N}(\mu_z, \sigma_z^2)$ is the encoder distribution
- $p(z) = \mathcal{N}(0, 1)$ is the prior distribution
- $D_{KL}$ is the Kullback-Leibler divergence

### 8.3 GAN Autoencoder (GAN-AE)

The objective function for GAN-AE:

$\psi = \sum_j [x[j] \log z_0[j] + (1-x[j]) \log(1-z_0[j])] + [\log(D(z_r)) + (1-\log(D(z_f)))]$

Where:
- $D$ is the discriminator network
- $z_r$ is a sample from the prior distribution
- $z_f$ is a sample from the encoder distribution

## 9. Computational Complexity

### 9.1 Autoencoder Complexity

For an autoencoder with $L$ layers:
- Forward pass: $\mathcal{O}(L)$ matrix multiplications
- Backward pass: $\mathcal{O}(L)$ matrix multiplications
- Total per sample: $\mathcal{O}(2L)$ matrix multiplications

### 9.2 NGC Model Complexity

For an NGC model with $L$ layers and $T$ inference iterations:
- Forward pass (predictions): $\mathcal{O}(L \times T)$ matrix multiplications
- Error computation: $\mathcal{O}(L \times T)$ element-wise operations
- State updates: $\mathcal{O}(L \times T)$ matrix multiplications
- Total per sample: $\mathcal{O}(2L \times T)$ matrix multiplications

## 10. Feature Analysis Mathematics

### 10.1 Feature Composition

The output of an NGC model can be expressed as a weighted sum of features:

$z_0 = \sum_{j=1}^{J_1} z_1[j] \cdot W_1[:,j]$

Where:
- $z_1[j]$ is the activation of neuron $j$ in layer 1
- $W_1[:,j]$ is the weight vector connecting neuron $j$ to the output layer

### 10.2 Hierarchical Feature Control

The activation of neurons in layer 1 is controlled by higher layers:

$z_1 = g_1(W_2 \cdot \Phi_2(z_2) + V_1 \cdot \Phi_1(z_1))$

Where the term $W_2 \cdot \Phi_2(z_2)$ represents the top-down control from layer 2, and $V_1 \cdot \Phi_1(z_1)$ represents lateral interactions within layer 1.
