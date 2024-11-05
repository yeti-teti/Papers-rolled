import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F


# Set random seed for reproducibility
torch.manual_seed(0)

class AdaptiveMixtureOfGaussians(nn.Module):
    def __init__(self, num_components=5, init_std=0.1):
        """
        Rather than using a single Gaussian, it is possible to use a mixture of several Gaussians whose means, variances, and mixing proportions are adapted as the network is trained
        """
        super(AdaptiveMixtureOfGaussians, self).__init__()
        self.init_std = init_std
        self.num_components = num_components
        self.means = nn.Parameter(torch.randn(num_components) * init_std)
        self.variances = nn.Parameter(torch.ones(num_components) * init_std)
        self.mixing_coeffs = nn.Parameter(torch.ones(num_components) / num_components)


    @property
    def variances(self):
        # Apply softplus to ensure variances are positive
        return F.softplus(self.raw_variances) + 1e-6  # Adding epsilon to avoid zero

    def sample(self):
        """
        To communicate a set of noisy weights, the sender first collapses the posterior probability distribution for each weight by using a source of random bits to pick a precise value for the weight.
        """
        # Adds controlled noise to the weights. Simulating a situation where weights are not perfectly precise but are drawn from a probability distribution. Sampling allows the model to explore different weight configurations during training, which prevents overfitting and promotes a more flexible representation.
        
        # Select a component based on the mixing coefficients
        component = torch.multinomial(torch.softmax(self.mixing_coeffs, dim=0), 1).item()
        # Compute standard deviation
        std = self.variances[component].sqrt()
        # Sample from the normal distribution
        return torch.normal(self.means[component], std)

    
    def log_prob(self, weight):
        # Flatten the weight tensor to 1D
        weight_flat = weight.view(-1)  # Shape: (N,)

        # Expand dimensions to enable broadcasting
        weight_expanded = weight_flat.unsqueeze(1)  # Shape: (N, 1)
        means_expanded = self.means.unsqueeze(0)    # Shape: (1, K)
        variances_expanded = self.variances.unsqueeze(0)  # Shape: (1, K)

        # Compute the differences and log probabilities
        diffs = weight_expanded - means_expanded  # Shape: (N, K)
        log_probs = -0.5 * (diffs ** 2) / variances_expanded
        log_probs -= torch.log(variances_expanded.sqrt() * np.sqrt(2 * np.pi))

        # Add log mixing coefficients
        log_mixing_coeffs = torch.log_softmax(self.mixing_coeffs, dim=0)  # Shape: (K,)
        log_probs += log_mixing_coeffs.unsqueeze(0)  # Shape: (N, K)

        # LogSumExp over the mixture components
        log_prob_weights = torch.logsumexp(log_probs, dim=1)  # Shape: (N,)

        # Sum over all weights to get the total log probability
        total_log_prob = log_prob_weights.sum()  # Scalar

        return total_log_prob

    
    def reinitialize_small_components(self, threshold=1e-2):
        """
        "If some components in the mixture become unused or approach zero probability, they can be reinitialized to prevent becoming irrelevant."
        """
        for i in range(self.num_components):
            if torch.softmax(self.mixing_coeffs, dim=0)[i] < threshold:
                self.means[i].data = torch.randn(1) * 0.1
                self.variances[i].data = torch.ones(1) * 0.1
                self.mixing_coeffs[i].data = torch.ones(1) / self.num_components

class MDLNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, mixture_model_hidden, mixture_model_output, noise_anneal_schedule=None):
        super(MDLNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.mixture_model_hidden = mixture_model_hidden
        self.mixture_model_output = mixture_model_output
        self.noise_anneal_schedule = noise_anneal_schedule or (lambda epoch: 0.1)
        
    def forward(self, x):
        # Ensure input dimensions are correct
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x

    def noisy_weights(self, epoch):
        """
        "A standard way of limiting the amount of information in a number is to add zero-mean Gaussian noise."
        "An alternative approach is to start with a multivariate Gaussian distribution over weight vectors and to change both the mean and the variance of this cloud of weight vectors so as to reduce some cost function"
        """
        # Adds noise to each weight during the forward pass, controlled by a noise annealing schedule. The annealing schedule reduces the noise level over time, starting with higher noise (to allow exploration and prevent overfitting) and gradually lowering it (to allow the model to converge to a more precise solution). Annealing schedule here reflects that goal by allowing high variance initially, which smoothens over time as the model converges.
        
        
        noise_level = self.noise_anneal_schedule(epoch)
    
        # For hidden layer
        for name, param in self.hidden.named_parameters():
            if 'weight' in name:
                noise = torch.randn_like(param) * self.mixture_model_hidden.sample() * noise_level
                param.data.add_(noise)
            elif 'bias' in name:
                # Optionally add noise or skip
                pass
        
        # For output layer
        for name, param in self.output.named_parameters():
            if 'weight' in name:
                noise = torch.randn_like(param) * self.mixture_model_output.sample() * noise_level
                param.data.add_(noise)
            elif 'bias' in name:
                # Optionally add noise or skip
                pass



# Monte Carlo-based Expected Squared Error Calculation
def monte_carlo_expected_error(model, inputs, targets, num_samples=10):
    """
    "For general feedforward networks with noisy weights, the expected squared errors are not easy to compute."
    "Fortunately, if there is only one hidden layer and if the output units are linear, it is possible to compute the expected squared error exactly."
    """
    # The Monte Carlo simulation here calculates the expected error by averaging over multiple forward passes with noisy weights.
    total_loss = 0.0
    for _ in range(num_samples):
        predictions = model(inputs)
        total_loss += torch.sum((predictions - targets) ** 2)
    return total_loss / (2 * num_samples)


# MDL Loss Function Combining Data Misfit and Weight Complexity
# MDL Loss with adaptive precision
def mdl_loss(inputs, targets, model, data_variance, weight_variance):
    """
    "The Minimum Description Length Principle asserts that the best model is the one that minimizes the combined cost of describing the model and describing the misfit between the model and the data."
    "The total description length is minimized by minimizing the sum of two terms: (1) the squared error, and (2) the weight complexity penalty based on the posterior distribution of the weights and their coding priors."
    """
    """
    This function directly implements the MDL principle, which minimizes both the error and the complexity. The weight complexity term encourages the model to use simpler weight configurations by penalizing complex or information-rich weights, directly enforcing the goal of minimizing the amount of information encoded in the model.
    """
    data_misfit = monte_carlo_expected_error(model, inputs, targets) / data_variance
    weight_complexity = 0

    # For hidden layer
    for param in model.hidden.parameters():
        kl_div = torch.sum(param ** 2) / (2 * weight_variance) - model.mixture_model_hidden.log_prob(param)
        weight_complexity += kl_div.sum()
    
    # For output layer
    for param in model.output.parameters():
        kl_div = torch.sum(param ** 2) / (2 * weight_variance) - model.mixture_model_output.log_prob(param)
        weight_complexity += kl_div.sum()
    
    return data_misfit + weight_complexity

# Generate synthetic dataset
def generate_data(num_samples, input_size):
    X = torch.randn(num_samples, input_size)
    y = (X.sum(dim=1) + torch.randn(num_samples) * 0.1).unsqueeze(1)
    return X, y


# Training with Dynamic Variance Adjustment and Early Stopping
# Training function with adaptive data and weight variance adjustment
def train(model, dataloader, optimizer, data_variance, weight_variance, mixture_model_hidden, mixture_model_output, 
          data_variance_optimizer, weight_variance_optimizer, mixture_optimizer_hidden, mixture_optimizer_output, num_epochs=100):
    """
    "The variance of the noise added to the weights and the coding priors are optimized during the learning process. This allows the precision of the weights to be dynamically adjusted based on the trade-off between the cost of encoding the weights and the data misfit."
    """
    """
    The training function dynamically adjusts data_variance and weight_variance using separate optimizers, allowing these hyperparameters to adapt as training progresses. This adaptive adjustment helps to learn the variance parameters during training to balance data fit and generalization. Additionally, early stopping is applied to prevent overfitting, halting training when no improvement is observed, which aligns with the MDL principle objective of finding a minimal and effective description.
    """

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            data_variance_optimizer.zero_grad()
            weight_variance_optimizer.zero_grad()
            mixture_optimizer_hidden.zero_grad()
            mixture_optimizer_output.zero_grad()

            model.noisy_weights(epoch)
            outputs = model(inputs)
            loss = mdl_loss(inputs, targets, model, data_variance, weight_variance)

            loss.backward()
            optimizer.step()
            data_variance_optimizer.step()
            weight_variance_optimizer.step()
            mixture_optimizer_hidden.step()  
            mixture_optimizer_output.step()

        if epoch % 10 == 0:
            mixture_model_hidden.reinitialize_small_components()
            mixture_model_output.reinitialize_small_components()

            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
            print(f"Hidden Layer Mixture Means: {mixture_model_hidden.means.detach().numpy()}")
            print(f"Hidden Layer Mixture Variances: {mixture_model_hidden.variances.detach().numpy()}")
            print(f"Output Layer Mixture Means: {mixture_model_output.means.detach().numpy()}")
            print(f"Output Layer Mixture Variances: {mixture_model_output.variances.detach().numpy()}")

        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping due to no improvement.")
                return
        

# Noise annealing schedule function
def noise_schedule(epoch, start=0.01, end=0.1, epochs=100):
    return start + (end - start) * (epoch / epochs)


# Main configuration and training
input_size = 128
hidden_size = 4
output_size = 1
num_samples = 100
data_variance = nn.Parameter(torch.tensor(0.1), requires_grad=True)
weight_variance = nn.Parameter(torch.tensor(0.01), requires_grad=True)
adaptive_mixture_hidden = AdaptiveMixtureOfGaussians(num_components=5, init_std=0.1)
adaptive_mixture_output = AdaptiveMixtureOfGaussians(num_components=5, init_std=0.1)

# Instantiate model, dataset, and optimizers
model = MDLNet(input_size, hidden_size, output_size, adaptive_mixture_hidden, adaptive_mixture_output, 
               noise_anneal_schedule=lambda epoch: noise_schedule(epoch))

X, y = generate_data(num_samples, input_size)
print("Shape of X:", X.shape)  # Should be (num_samples, 128)
print("Shape of y:", y.shape)  # Should be (num_samples, 1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)
data_variance_optimizer = optim.SGD([data_variance], lr=0.001)
weight_variance_optimizer = optim.SGD([weight_variance], lr=0.001)
mixture_optimizer_hidden = optim.SGD([
    model.mixture_model_hidden.means,
    model.mixture_model_hidden.variances,
    model.mixture_model_hidden.mixing_coeffs
], lr=0.01)

mixture_optimizer_output = optim.SGD([
    model.mixture_model_output.means,
    model.mixture_model_output.variances,
    model.mixture_model_output.mixing_coeffs
], lr=0.01)


# Train the model
train(model, dataloader, optimizer, data_variance, weight_variance, adaptive_mixture_hidden, adaptive_mixture_output, 
      data_variance_optimizer, weight_variance_optimizer, mixture_optimizer_hidden, mixture_optimizer_output, num_epochs=100)