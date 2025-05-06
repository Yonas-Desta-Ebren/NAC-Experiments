# Neural Generative Coding (NGC) Tutorial

This directory contains resources for understanding and experimenting with Neural Generative Coding (NGC) models, as described in the paper "Neural Generative Coding Through Probabilistic Feedback" (2012.03405v4).

## Contents

1. **NGC_Documentation.md**: A comprehensive guide to NGC models, including theoretical framework, mathematical formulation, model variants, metaphorical examples, and practical applications.

2. **NGC_Simulation.py**: A Python script that implements simplified versions of the NGC model variants (GNCN-PDH, GNCN-t1, GNCN-t1-Sigma) and demonstrates their behavior on a toy dataset.

3. **model_summary.md**: A summary of the stable versions of the NGC models that we've run on the MNIST dataset.

4. **interview_preparation.md**: Detailed mathematical explanations and interview questions/answers about NGC models and related generative models.

5. **model_comparison.md**: A comprehensive comparison of NGC models with other generative models like VAEs and RAEs.

## Getting Started

### Prerequisites

To run the simulation, you'll need:

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn

You can install the required packages using pip:

```bash
pip install numpy matplotlib scikit-learn
```

### Running the Simulation

To run the NGC simulation:

```bash
python NGC_Simulation.py
```

This will:
1. Generate a simple 2D dataset (two moons)
2. Train three NGC model variants (GNCN-PDH, GNCN-t1, GNCN-t1-Sigma)
3. Plot training curves, reconstructions, and latent space visualizations
4. Compare the performance of the three models

## Understanding NGC Models

### Key Concepts

1. **Predictive Coding**: NGC is based on the predictive coding theory from neuroscience, which posits that the brain is constantly trying to predict its sensory inputs and updates its internal models based on prediction errors.

2. **Hierarchical Structure**: NGC models have a hierarchical structure where each layer tries to predict the activity of the layer below it.

3. **Local Learning Rules**: NGC uses local learning rules based on prediction errors, making it more biologically plausible than backpropagation.

4. **Iterative Settling**: NGC involves an iterative settling process where the network converges to a stable state before weight updates are applied.

### Model Variants

1. **GNCN-PDH (Generative Neural Coding Network with Predictive Discrete Hebbian learning)**:
   - Uses Predictive Discrete Hebbian learning
   - Activation function: tanh
   - Output function: sigmoid

2. **GNCN-t1 (Generative Neural Coding Network - Type 1)**:
   - A variant of NGC with a different architecture
   - Activation function: tanh
   - Output function: sigmoid

3. **GNCN-t1-Sigma (Generative Neural Coding Network - Type 1 with Sigma)**:
   - Extends GNCN-t1 by including variance parameters
   - Activation function: relu
   - Output function: sigmoid

## Extending the Simulation

The simulation code is designed to be modular and extensible. Here are some ways you can extend it:

1. **Try different datasets**: Modify the code to use different datasets, such as MNIST or Fashion-MNIST.

2. **Experiment with hyperparameters**: Adjust parameters like beta, K, learning rate, etc., to see how they affect performance.

3. **Add more model variants**: Implement other NGC variants or compare with traditional models like VAEs.

4. **Visualize the iterative settling process**: Add code to visualize how the state variables evolve during the iterative settling process.

5. **Implement more complex architectures**: Add convolutional layers or recurrent connections to handle more complex data.

## References

1. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 2(1), 79-87.

2. Whittington, J. C., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural computation, 29(5), 1229-1262.

3. Millidge, B., Tschantz, A., & Buckley, C. L. (2020). Predictive coding approximates backprop along arbitrary computation graphs. arXiv preprint arXiv:2006.04182.

4. Salvatori, T., Song, Y., Hong, Y., Sha, L., Frieder, S., Xu, Z., ... & Bogacz, R. (2021). Neural generative coding through probabilistic feedback. arXiv preprint arXiv:2012.03405v4.
