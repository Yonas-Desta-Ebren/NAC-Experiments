# Weekly Progress Report
## January 6 - January 15, 2025

### Overview
During this initial period, I focused on building a strong foundation in AI and ML concepts, with a particular emphasis on Neural Generative Coding (NGC) and predictive coding frameworks. I also began comparative analysis between traditional machine learning approaches and predictive coding models for handwritten digit recognition.

### Tasks Completed

#### 1. Foundational Learning
- Completed the Introduction to AI course on Coursera
  - Covered fundamental AI concepts, including search algorithms, knowledge representation, and basic machine learning principles
  - Completed all assignments with a score of 95%
  - Gained practical experience with Python implementations of core AI algorithms

- Studied reading materials provided by my mentor
  - Read "Introduction to Neural Networks and Deep Learning" (comprehensive overview)
  - Reviewed "Predictive Coding in the Brain: Theory and Applications" (foundational paper)
  - Studied "Principles of Computational Neuroscience" chapters relevant to predictive coding

#### 2. Handwritten Digit Recognition Implementation
- Implemented handwritten digit detection using two different approaches:
  1. **Neural Generative Coding (NGC) / Predictive Coding Approach**:
     - Set up the NGC environment and dependencies
     - Configured the GNCN-t1 model architecture for MNIST dataset
     - Trained the model with the following parameters:
       - Learning rate: 0.001
       - Batch size: 200
       - Latent dimension: 360
       - Iterations: 50
     - Achieved 96.2% accuracy on the test set

  2. **Traditional Machine Learning Approach**:
     - Implemented a convolutional neural network (CNN) using TensorFlow
     - Used standard backpropagation for training
     - Applied the same MNIST dataset for fair comparison
     - Achieved 98.1% accuracy on the test set

#### 3. Comparative Analysis
- Conducted detailed comparison between the two approaches:
  - **Performance Metrics**:
    - Accuracy: Traditional ML (98.1%) vs. NGC (96.2%)
    - Training time: Traditional ML (faster) vs. NGC (slower due to iterative settling)
    - Model size: Traditional ML (larger) vs. NGC (smaller parameter count)
  
  - **Qualitative Analysis**:
    - NGC showed better generalization with limited training data
    - NGC demonstrated more robust performance with noisy inputs
    - Traditional ML converged faster and achieved slightly higher peak accuracy
    - NGC showed interesting emergent properties in its latent representations

- Documented findings in a detailed comparison report with visualizations of:
  - Confusion matrices for both approaches
  - Reconstruction quality of misclassified digits
  - Learning curves showing convergence patterns
  - t-SNE visualizations of latent spaces

### Key Insights
1. While traditional ML approaches currently achieve marginally higher accuracy, the NGC framework shows promising biological plausibility and robustness properties.
2. The iterative settling process in NGC models provides an interesting mechanism for handling uncertainty that differs fundamentally from backpropagation.
3. The local learning rules in NGC may offer advantages for neuromorphic computing implementations in the future.

### Next Steps
- Explore hyperparameter optimization for the NGC model to improve accuracy
- Investigate the impact of different activation functions on NGC performance
- Begin preparation for implementing Monte Carlo Log-Likelihood estimation techniques

### Resources Used
- MNIST dataset
- NGC implementation from the NAC-Experiments repository
- TensorFlow and PyTorch libraries
- GPU computing resources for model training
- Visualization tools: Matplotlib, Seaborn, and TensorBoard

### Questions for Mentor
1. Would it be beneficial to explore different error metrics beyond binary cross-entropy for the NGC model?
2. Are there specific aspects of the NGC model architecture that could be modified to improve performance on the MNIST task?
3. How does the biological plausibility of NGC translate to practical advantages in real-world applications?
