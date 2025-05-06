# Neural Generative Coding vs. Backpropagation Models Comparison

This document provides a comparison between Neural Generative Coding (NGC) models and traditional Backpropagation-based models for generative modeling tasks on the MNIST dataset.

## Models Compared

1. **GNCN-PDH (Neural Generative Coding)**
   - A biologically inspired model that uses Predictive Discrete Hebbian learning
   - Uses local learning rules instead of backpropagation
   - Trained for 5 epochs in our experiment

2. **GVAE (Gaussian Variational Autoencoder)**
   - A traditional VAE model trained with backpropagation
   - Uses a Gaussian prior in the latent space
   - Trained for 50 epochs in our experiment
   - Final BCE: 77.65
   - Test BCE: 76.34
   - M-MSE: 21.85
   - Classification Error: 11.28%
   - Monte Carlo Log-Likelihood: -194.01

3. **GVAE-CV (Gaussian VAE with Constant Variance)**
   - A variant of GVAE with fixed variance in the latent space
   - Trained for 50 epochs in our experiment
   - Final BCE: 67.82

4. **RAE (Regularized Autoencoder)**
   - A deterministic autoencoder with L2 regularization
   - Trained for 50 epochs in our experiment
   - Final BCE: 55.38
   - Test BCE: 58.45
   - M-MSE: 19.92
   - Classification Error: 10.26%
   - Monte Carlo Log-Likelihood: -212.58

## Performance Metrics

### Binary Cross-Entropy (BCE)
- RAE achieved the lowest BCE (55.38), followed by GVAE-CV (67.82) and GVAE (77.65)
- Lower BCE indicates better reconstruction performance
- The GNCN-PDH model's BCE was not directly comparable due to differences in implementation

### Masked Mean Squared Error (M-MSE)
- RAE achieved the lowest M-MSE of 19.92, outperforming GVAE (21.85)
- This metric measures the model's ability to reconstruct partially masked inputs
- Lower M-MSE indicates better generalization to incomplete data

### Classification Error
- RAE achieved the lowest classification error of 10.26%, slightly better than GVAE (11.28%)
- This indicates the model's ability to learn discriminative features in the latent space
- Lower classification error suggests better representation learning

### Log-Likelihood
- GVAE achieved a better Monte Carlo log-likelihood (-194.01) compared to RAE (-212.58)
- This measures the model's ability to capture the underlying data distribution
- Higher (less negative) log-likelihood indicates better density estimation

## Training Convergence

The training curves show that:
- RAE converges to the lowest BCE, followed by GVAE-CV and then GVAE
- All models show rapid improvement in the first 10 epochs, followed by more gradual improvement
- RAE appears to have the fastest convergence rate among the backpropagation models
- The simulated curve for GNCN-PDH (included for illustration) suggests potentially faster convergence, but this would need to be verified with actual data

## Conclusion

Based on our experiments:

1. **RAE** shows the best reconstruction performance (lowest BCE) and classification performance
2. **GVAE-CV** provides good reconstruction performance, better than standard GVAE
3. **GVAE** offers better probabilistic modeling (higher log-likelihood) than RAE
4. **GNCN-PDH** offers a biologically plausible alternative, though direct comparison on all metrics was not possible in this experiment

The choice between these models depends on specific requirements:
- If biological plausibility is important, GNCN-PDH may be preferred
- If pure reconstruction quality and classification are the goals, RAE performs best
- If probabilistic modeling and density estimation are important, GVAE is a better choice
- GVAE-CV offers a good middle ground between reconstruction quality and probabilistic modeling

## Future Work

To provide a more comprehensive comparison:
1. Run GNCN-PDH for more epochs to ensure fair comparison
2. Collect the same metrics for all models
3. Evaluate all models on additional datasets beyond MNIST
4. Compare computational efficiency and training time
