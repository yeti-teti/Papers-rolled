import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)

class AdaptiveMixtureOfGaussians(nn.Module):
    def __init__(self, num_components=5, init_std=0.1):
        super(AdaptiveMixtureOfGaussians, self).__init__()
        self.num_components = num_components
        self.means = nn.Parameter(torch.randn(num_components) * init_std)
        self.variances = nn.Parameter(torch.ones(num_components) * init_std)
        self.mixing_coeffs = nn.Parameter(torch.ones(num_components) / num_components)

    def sample(self):
        component = torch.multinomial(torch.softmax(self.mixing_coeffs, dim=0), 1).item()
        return torch.normal(self.means[component], self.variances[component].sqrt())
    
    def log_prob(self, weight):
        log_probs = -0.5 * ((weight - self.means) ** 2) / self.variances - torch.log(self.variances.sqrt() * np.sqrt(2 * np.pi))
        return torch.logsumexp(log_probs + torch.log_softmax(self.mixing_coeffs, dim=0), dim=0)
    
    def reinitialize_small_components(self, threshold=1e-2):
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
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x

    def noisy_weights(self, epoch):
        noise_level = self.noise_anneal_schedule(epoch)
        for param, mixture_model in zip(self.parameters(), [self.mixture_model_hidden, self.mixture_model_output]):
            noise = torch.stack([mixture_model.sample() * noise_level for _ in range(param.numel())])
            param.data.add_(noise.view(param.shape))

# Monte Carlo-based Expected Squared Error Calculation
def monte_carlo_expected_error(model, inputs, targets, num_samples=10):
    total_loss = 0.0
    for _ in range(num_samples):
        predictions = model(inputs)
        total_loss += torch.sum((predictions - targets) ** 2)
    return total_loss / (2 * num_samples)

# MDL Loss with adaptive precision
def mdl_loss(predictions, targets, model, data_variance, weight_variance, mixture_model_hidden, mixture_model_output):
    data_misfit = monte_carlo_expected_error(model, predictions, targets) / data_variance
    weight_complexity = 0

    for param, mixture_model in zip(model.parameters(), [mixture_model_hidden, mixture_model_output]):
        kl_div = torch.sum(param ** 2) / (2 * weight_variance) - mixture_model.log_prob(param)
        weight_complexity += kl_div.sum()
    
    return data_misfit + weight_complexity

# Generate synthetic dataset
def generate_data(num_samples, input_size):
    X = torch.randn(num_samples, input_size)
    y = (X.sum(dim=1) + torch.randn(num_samples) * 0.1).unsqueeze(1)
    return X, y

# Training function with adaptive data and weight variance adjustment
def train(model, dataloader, optimizer, data_variance, weight_variance, mixture_model_hidden, mixture_model_output, 
          data_variance_optimizer, weight_variance_optimizer, num_epochs=100):
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            data_variance_optimizer.zero_grad()
            weight_variance_optimizer.zero_grad()
            
            model.noisy_weights(epoch)
            outputs = model(inputs)
            loss = mdl_loss(outputs, targets, model, data_variance, weight_variance, mixture_model_hidden, mixture_model_output)
            
            loss.backward()
            optimizer.step()
            data_variance_optimizer.step()
            weight_variance_optimizer.step()

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
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)
data_variance_optimizer = optim.SGD([data_variance], lr=0.001)
weight_variance_optimizer = optim.SGD([weight_variance], lr=0.001)
mixture_optimizer_hidden = optim.SGD([adaptive_mixture_hidden.means, adaptive_mixture_hidden.variances, adaptive_mixture_hidden.mixing_coeffs], lr=0.01)
mixture_optimizer_output = optim.SGD([adaptive_mixture_output.means, adaptive_mixture_output.variances, adaptive_mixture_output.mixing_coeffs], lr=0.01)

# Train the model
train(model, dataloader, optimizer, data_variance, weight_variance, adaptive_mixture_hidden, adaptive_mixture_output, 
      data_variance_optimizer, weight_variance_optimizer, num_epochs=100)
