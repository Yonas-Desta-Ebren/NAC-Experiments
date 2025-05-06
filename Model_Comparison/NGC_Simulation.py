"""
Neural Generative Coding (NGC) Simulation

This script provides a simple implementation of the Neural Generative Coding (NGC) model
and its variants (GNCN-PDH, GNCN-t1, GNCN-t1-Sigma) for educational purposes.

The simulation demonstrates the key concepts of NGC on a simple toy dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import time

# Set random seed for reproducibility
np.random.seed(42)

# Create a simple 2D dataset (two moons)
X, y = make_moons(n_samples=1000, noise=0.1)
# Normalize the data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split into training and test sets
X_train = X[:800]
X_test = X[800:]
y_train = y[:800]
y_test = y[800:]

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class NGCBase:
    """Base class for Neural Generative Coding models."""
    
    def __init__(self, layer_sizes, activation='tanh', output_activation='sigmoid',
                 beta=0.1, K=50, leak=0.001, lambda_reg=0.01, learning_rate=0.001):
        """
        Initialize the NGC model.
        
        Parameters:
        -----------
        layer_sizes : list
            List of layer sizes (including input and output layers)
        activation : str
            Activation function for hidden layers ('tanh' or 'relu')
        output_activation : str
            Activation function for the output layer ('sigmoid' or 'linear')
        beta : float
            Controls the state update rate
        K : int
            Number of steps in the iterative settling process
        leak : float
            Leak parameter for state variables
        lambda_reg : float
            Weight decay parameter
        learning_rate : float
            Learning rate for weight updates
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.beta = beta
        self.K = K
        self.leak = leak
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        
        # Set activation functions
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        if output_activation == 'sigmoid':
            self.output_activation = sigmoid
            self.output_activation_derivative = sigmoid_derivative
        elif output_activation == 'linear':
            self.output_activation = lambda x: x
            self.output_activation_derivative = lambda x: 1
        else:
            raise ValueError(f"Unsupported output activation function: {output_activation}")
        
        # Initialize weights and state variables
        self.weights = []
        for i in range(self.num_layers - 1):
            # Initialize weights with small random values
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
            self.weights.append(W)
        
        # Initialize state variables
        self.state_variables = [np.zeros(size) for size in layer_sizes]
        
        # For tracking training progress
        self.train_losses = []
        self.test_losses = []
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : ndarray
            Input data
            
        Returns:
        --------
        list
            Activations of all layers
        """
        activations = [x]
        for i in range(self.num_layers - 1):
            if i < self.num_layers - 2:
                # Hidden layers
                z = np.dot(self.weights[i], activations[-1])
                a = self.activation(z)
            else:
                # Output layer
                z = np.dot(self.weights[i], activations[-1])
                a = self.output_activation(z)
            activations.append(a)
        return activations
    
    def compute_prediction_errors(self, activations):
        """
        Compute prediction errors for all layers.
        
        Parameters:
        -----------
        activations : list
            Activations of all layers
            
        Returns:
        --------
        list
            Prediction errors for all layers
        """
        errors = []
        for i in range(self.num_layers - 1):
            # Compute prediction of layer i from layer i+1
            prediction = np.dot(self.weights[i].T, activations[i+1])
            # Compute prediction error
            error = activations[i] - prediction
            errors.append(error)
        return errors
    
    def update_state_variables(self, activations, errors):
        """
        Update state variables based on prediction errors.
        
        Parameters:
        -----------
        activations : list
            Activations of all layers
        errors : list
            Prediction errors for all layers
            
        Returns:
        --------
        list
            Updated activations
        """
        new_activations = activations.copy()
        
        # Update state variables for hidden layers
        for i in range(1, self.num_layers - 1):
            # Bottom-up error
            bottom_up = np.dot(self.weights[i-1], errors[i-1])
            
            # Top-down error (if not the top layer)
            if i < self.num_layers - 1:
                top_down = errors[i]
            else:
                top_down = 0
            
            # Update state variable
            delta = self.beta * (bottom_up - top_down) - self.leak * new_activations[i]
            new_activations[i] += delta
            
            # Apply activation function
            if i < self.num_layers - 1:
                new_activations[i] = self.activation(new_activations[i])
            else:
                new_activations[i] = self.output_activation(new_activations[i])
        
        return new_activations
    
    def update_weights(self, activations, errors):
        """
        Update weights based on prediction errors.
        
        Parameters:
        -----------
        activations : list
            Activations of all layers
        errors : list
            Prediction errors for all layers
        """
        for i in range(self.num_layers - 1):
            # Hebbian-like weight update
            delta_W = self.learning_rate * np.outer(activations[i+1], errors[i])
            # Weight decay
            delta_W -= self.lambda_reg * self.weights[i]
            # Update weights
            self.weights[i] += delta_W
    
    def iterative_settling(self, x):
        """
        Perform the iterative settling process.
        
        Parameters:
        -----------
        x : ndarray
            Input data
            
        Returns:
        --------
        list
            Final activations after settling
        """
        # Initialize activations with input
        activations = self.forward(x)
        
        # Iterative settling process
        for _ in range(self.K):
            # Compute prediction errors
            errors = self.compute_prediction_errors(activations)
            # Update state variables
            activations = self.update_state_variables(activations, errors)
        
        return activations
    
    def train_step(self, x):
        """
        Perform one training step on a single input.
        
        Parameters:
        -----------
        x : ndarray
            Input data
            
        Returns:
        --------
        float
            Reconstruction error
        """
        # Iterative settling
        activations = self.iterative_settling(x)
        
        # Compute prediction errors
        errors = self.compute_prediction_errors(activations)
        
        # Update weights
        self.update_weights(activations, errors)
        
        # Compute reconstruction error
        reconstruction = activations[-1]
        reconstruction_error = np.mean((x - reconstruction)**2)
        
        return reconstruction_error
    
    def train(self, X, X_test=None, epochs=100, batch_size=32, verbose=True):
        """
        Train the model.
        
        Parameters:
        -----------
        X : ndarray
            Training data
        X_test : ndarray, optional
            Test data for evaluation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        verbose : bool
            Whether to print progress
        """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            # Train on batches
            epoch_loss = 0
            for i in range(n_batches):
                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size
                X_batch = X_shuffled[batch_start:batch_end]
                
                batch_loss = 0
                for x in X_batch:
                    batch_loss += self.train_step(x)
                
                epoch_loss += batch_loss / len(X_batch)
            
            epoch_loss /= n_batches
            self.train_losses.append(epoch_loss)
            
            # Evaluate on test data if provided
            if X_test is not None:
                test_loss = self.evaluate(X_test)
                self.test_losses.append(test_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
    
    def evaluate(self, X):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X : ndarray
            Test data
            
        Returns:
        --------
        float
            Average reconstruction error
        """
        total_loss = 0
        for x in X:
            activations = self.iterative_settling(x)
            reconstruction = activations[-1]
            loss = np.mean((x - reconstruction)**2)
            total_loss += loss
        
        return total_loss / len(X)
    
    def reconstruct(self, X):
        """
        Reconstruct the input data.
        
        Parameters:
        -----------
        X : ndarray
            Input data
            
        Returns:
        --------
        ndarray
            Reconstructed data
        """
        reconstructions = []
        for x in X:
            activations = self.iterative_settling(x)
            reconstruction = activations[-1]
            reconstructions.append(reconstruction)
        
        return np.array(reconstructions)
    
    def plot_training_curve(self):
        """Plot the training and test loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.test_losses:
            plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('Training Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_reconstructions(self, X, n_samples=5):
        """
        Plot original and reconstructed samples.
        
        Parameters:
        -----------
        X : ndarray
            Input data
        n_samples : int
            Number of samples to plot
        """
        # Select random samples
        indices = np.random.choice(len(X), n_samples, replace=False)
        samples = X[indices]
        
        # Reconstruct samples
        reconstructions = self.reconstruct(samples)
        
        # Plot
        plt.figure(figsize=(12, 4))
        for i in range(n_samples):
            # Original
            plt.subplot(2, n_samples, i + 1)
            plt.scatter(samples[i, 0], samples[i, 1], color='blue')
            plt.title(f"Original {i+1}")
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            
            # Reconstruction
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.scatter(reconstructions[i, 0], reconstructions[i, 1], color='red')
            plt.title(f"Reconstruction {i+1}")
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_latent_space(self, X, y=None):
        """
        Plot the latent space representation.
        
        Parameters:
        -----------
        X : ndarray
            Input data
        y : ndarray, optional
            Labels for coloring the points
        """
        # Get latent representations
        latent_reps = []
        for x in X:
            activations = self.iterative_settling(x)
            latent_rep = activations[1]  # Assuming the first hidden layer is the latent space
            latent_reps.append(latent_rep)
        
        latent_reps = np.array(latent_reps)
        
        # Plot
        plt.figure(figsize=(10, 8))
        if y is not None:
            for label in np.unique(y):
                mask = y == label
                plt.scatter(latent_reps[mask, 0], latent_reps[mask, 1], label=f"Class {label}")
            plt.legend()
        else:
            plt.scatter(latent_reps[:, 0], latent_reps[:, 1])
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Representation')
        plt.grid(True)
        plt.show()


class GNCNPDH(NGCBase):
    """
    Generative Neural Coding Network with Predictive Discrete Hebbian learning (GNCN-PDH).
    
    This is a specific implementation of the NGC model with PDH learning.
    """
    
    def __init__(self, layer_sizes, **kwargs):
        # GNCN-PDH uses tanh activation by default
        kwargs.setdefault('activation', 'tanh')
        kwargs.setdefault('output_activation', 'sigmoid')
        super().__init__(layer_sizes, **kwargs)


class GNCNt1(NGCBase):
    """
    Generative Neural Coding Network - Type 1 (GNCN-t1).
    
    This is a variant of the NGC model with a different architecture.
    """
    
    def __init__(self, layer_sizes, **kwargs):
        # GNCN-t1 uses tanh activation by default
        kwargs.setdefault('activation', 'tanh')
        kwargs.setdefault('output_activation', 'sigmoid')
        super().__init__(layer_sizes, **kwargs)
    
    def update_state_variables(self, activations, errors):
        """
        Update state variables based on prediction errors.
        
        This implementation has a different connectivity pattern than the base NGC model.
        """
        new_activations = activations.copy()
        
        # Update state variables for hidden layers
        for i in range(1, self.num_layers - 1):
            # Bottom-up error
            bottom_up = np.dot(self.weights[i-1], errors[i-1])
            
            # Top-down error (if not the top layer)
            if i < self.num_layers - 1:
                top_down = np.dot(self.weights[i].T, errors[i])
            else:
                top_down = 0
            
            # Update state variable
            delta = self.beta * (bottom_up - top_down) - self.leak * new_activations[i]
            new_activations[i] += delta
            
            # Apply activation function
            if i < self.num_layers - 1:
                new_activations[i] = self.activation(new_activations[i])
            else:
                new_activations[i] = self.output_activation(new_activations[i])
        
        return new_activations


class GNCNt1Sigma(GNCNt1):
    """
    Generative Neural Coding Network - Type 1 with Sigma (GNCN-t1-Sigma).
    
    This extends GNCN-t1 by including variance parameters.
    """
    
    def __init__(self, layer_sizes, **kwargs):
        # GNCN-t1-Sigma uses relu activation by default
        kwargs.setdefault('activation', 'relu')
        kwargs.setdefault('output_activation', 'sigmoid')
        super().__init__(layer_sizes, **kwargs)
        
        # Initialize variance parameters
        self.sigma = [np.ones(size) for size in layer_sizes]
        
        # Minimum variance to prevent division by zero
        self.min_sigma = 0.01
    
    def compute_prediction_errors(self, activations):
        """
        Compute prediction errors for all layers, weighted by variance.
        """
        errors = []
        for i in range(self.num_layers - 1):
            # Compute prediction of layer i from layer i+1
            prediction = np.dot(self.weights[i].T, activations[i+1])
            # Compute prediction error
            error = activations[i] - prediction
            # Weight by variance
            weighted_error = error / np.maximum(self.sigma[i]**2, self.min_sigma)
            errors.append(weighted_error)
        return errors
    
    def update_state_variables(self, activations, errors):
        """
        Update state variables based on variance-weighted prediction errors.
        """
        new_activations = activations.copy()
        
        # Update state variables for hidden layers
        for i in range(1, self.num_layers - 1):
            # Bottom-up error
            bottom_up = np.dot(self.weights[i-1], errors[i-1])
            
            # Top-down error (if not the top layer)
            if i < self.num_layers - 1:
                top_down = np.dot(self.weights[i].T, errors[i])
            else:
                top_down = 0
            
            # Update state variable
            delta = self.beta * (bottom_up - top_down) - self.leak * new_activations[i]
            new_activations[i] += delta
            
            # Apply activation function
            if i < self.num_layers - 1:
                new_activations[i] = self.activation(new_activations[i])
            else:
                new_activations[i] = self.output_activation(new_activations[i])
        
        return new_activations
    
    def update_weights(self, activations, errors):
        """
        Update weights based on variance-weighted prediction errors.
        """
        for i in range(self.num_layers - 1):
            # Hebbian-like weight update with variance weighting
            delta_W = self.learning_rate * np.outer(activations[i+1], errors[i])
            # Weight decay
            delta_W -= self.lambda_reg * self.weights[i]
            # Update weights
            self.weights[i] += delta_W
    
    def update_sigma(self, activations, errors, learning_rate=0.01):
        """
        Update variance parameters.
        
        Parameters:
        -----------
        activations : list
            Activations of all layers
        errors : list
            Prediction errors for all layers
        learning_rate : float
            Learning rate for sigma updates
        """
        for i in range(self.num_layers - 1):
            # Compute squared prediction error
            squared_error = errors[i]**2 * self.sigma[i]**2
            
            # Update sigma
            delta_sigma = learning_rate * (squared_error - 1) * self.sigma[i]
            self.sigma[i] += delta_sigma
            
            # Ensure sigma stays positive
            self.sigma[i] = np.maximum(self.sigma[i], self.min_sigma)
    
    def train_step(self, x):
        """
        Perform one training step on a single input.
        """
        # Iterative settling
        activations = self.iterative_settling(x)
        
        # Compute prediction errors
        errors = self.compute_prediction_errors(activations)
        
        # Update weights
        self.update_weights(activations, errors)
        
        # Update sigma
        self.update_sigma(activations, errors)
        
        # Compute reconstruction error
        reconstruction = activations[-1]
        reconstruction_error = np.mean((x - reconstruction)**2)
        
        return reconstruction_error


def run_simulation():
    """Run the NGC simulation with all model variants."""
    
    print("Neural Generative Coding (NGC) Simulation")
    print("=========================================")
    
    # Define layer sizes (input, hidden, output)
    layer_sizes = [2, 5, 2]
    
    # Create and train GNCN-PDH model
    print("\nTraining GNCN-PDH model...")
    gncn_pdh = GNCNPDH(layer_sizes, beta=0.1, K=20, learning_rate=0.01)
    gncn_pdh.train(X_train, X_test, epochs=100, batch_size=32)
    
    # Create and train GNCN-t1 model
    print("\nTraining GNCN-t1 model...")
    gncn_t1 = GNCNt1(layer_sizes, beta=0.1, K=20, learning_rate=0.01)
    gncn_t1.train(X_train, X_test, epochs=100, batch_size=32)
    
    # Create and train GNCN-t1-Sigma model
    print("\nTraining GNCN-t1-Sigma model...")
    gncn_t1_sigma = GNCNt1Sigma(layer_sizes, beta=0.1, K=20, learning_rate=0.01)
    gncn_t1_sigma.train(X_train, X_test, epochs=100, batch_size=32)
    
    # Plot training curves
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(gncn_pdh.train_losses, label='Train')
    plt.plot(gncn_pdh.test_losses, label='Test')
    plt.title('GNCN-PDH')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(gncn_t1.train_losses, label='Train')
    plt.plot(gncn_t1.test_losses, label='Test')
    plt.title('GNCN-t1')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(gncn_t1_sigma.train_losses, label='Train')
    plt.plot(gncn_t1_sigma.test_losses, label='Test')
    plt.title('GNCN-t1-Sigma')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Plot reconstructions
    for model, name in [(gncn_pdh, 'GNCN-PDH'), (gncn_t1, 'GNCN-t1'), (gncn_t1_sigma, 'GNCN-t1-Sigma')]:
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'{name} Reconstructions')
        
        # Select random samples
        indices = np.random.choice(len(X_test), 5, replace=False)
        samples = X_test[indices]
        
        # Reconstruct samples
        reconstructions = model.reconstruct(samples)
        
        # Plot
        for i in range(5):
            # Original
            plt.subplot(2, 5, i + 1)
            plt.scatter(X_test[:, 0], X_test[:, 1], color='lightgray', alpha=0.3)
            plt.scatter(samples[i, 0], samples[i, 1], color='blue')
            plt.title(f"Original {i+1}")
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            
            # Reconstruction
            plt.subplot(2, 5, 5 + i + 1)
            plt.scatter(X_test[:, 0], X_test[:, 1], color='lightgray', alpha=0.3)
            plt.scatter(reconstructions[i, 0], reconstructions[i, 1], color='red')
            plt.title(f"Reconstruction {i+1}")
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig(f'{name}_reconstructions.png')
        plt.show()
    
    # Plot latent space
    for model, name in [(gncn_pdh, 'GNCN-PDH'), (gncn_t1, 'GNCN-t1'), (gncn_t1_sigma, 'GNCN-t1-Sigma')]:
        # Get latent representations
        latent_reps = []
        for x in X_test:
            activations = model.iterative_settling(x)
            latent_rep = activations[1]  # First hidden layer
            latent_reps.append(latent_rep)
        
        latent_reps = np.array(latent_reps)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_reps[y_test == 0, 0], latent_reps[y_test == 0, 1], label='Class 0')
        plt.scatter(latent_reps[y_test == 1, 0], latent_reps[y_test == 1, 1], label='Class 1')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title(f'{name} Latent Space')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{name}_latent_space.png')
        plt.show()
    
    # Compare final test losses
    final_losses = {
        'GNCN-PDH': gncn_pdh.test_losses[-1],
        'GNCN-t1': gncn_t1.test_losses[-1],
        'GNCN-t1-Sigma': gncn_t1_sigma.test_losses[-1]
    }
    
    print("\nFinal Test Losses:")
    for model, loss in final_losses.items():
        print(f"{model}: {loss:.4f}")
    
    # Plot final test losses
    plt.figure(figsize=(8, 6))
    plt.bar(final_losses.keys(), final_losses.values())
    plt.ylabel('Mean Squared Error')
    plt.title('Final Test Losses')
    plt.grid(True, axis='y')
    plt.savefig('final_test_losses.png')
    plt.show()


if __name__ == "__main__":
    run_simulation()
