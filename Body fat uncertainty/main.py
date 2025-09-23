import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# GPyTorch
from gpytorch.models import ExactGP
from gpytorch.means import LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
import kagglehub
import os
import gpytorch

# Download latest version
path = kagglehub.dataset_download("fedesoriano/body-fat-prediction-dataset")

print("Path to dataset files:", path)

# Read the CSV file from the downloaded dataset
# The dataset contains a file named "bodyfat.csv"
csv_path = os.path.join(path, "bodyfat.csv")
df = pd.read_csv(csv_path)
print(df.head())

# Typical columns in this dataset (yours may vary slightly):
# 'BodyFat', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip',
# 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist'
target_col = "BodyFat"

candidate_features = [
    "Age",
    "Weight",
    "Height",
    "Neck",
    "Chest",
    "Abdomen",
    "Hip",
    "Thigh",
    "Knee",
    "Ankle",
    "Biceps",
    "Forearm",
    "Wrist",
]

# Keep only available columns
features = [c for c in candidate_features if c in df.columns]

# Drop rows with missing target or features
df = df.dropna(subset=[target_col] + features).reset_index(drop=True)

X = df[features].values.astype(np.float64)
y = df[[target_col]].values.astype(np.float64)  # shape (n, 1)

# ===== Train/Test split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Convert to torch tensors
train_X = torch.tensor(X_train, dtype=torch.double)
train_Y = torch.tensor(y_train, dtype=torch.double)
test_X = torch.tensor(X_test, dtype=torch.double)
test_Y = torch.tensor(y_test, dtype=torch.double)

# Normalize inputs
X_min = train_X.min(0, keepdim=True)[0]
X_max = train_X.max(0, keepdim=True)[0]
train_X_scaled = (train_X - X_min) / (X_max - X_min)
test_X_scaled = (test_X - X_min) / (X_max - X_min)

# Standardize outputs
y_mean = train_Y.mean()
y_std = train_Y.std()
train_Y_scaled = (train_Y - y_mean) / y_std


# ===== Build GP model =====
class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(input_size=train_x.shape[1])
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# ===== Train GP =====
likelihood = GaussianLikelihood()
model = ExactGPModel(train_X_scaled, train_Y_scaled.squeeze(-1), likelihood)

model.double()
likelihood.double()

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
mll = ExactMarginalLogLikelihood(likelihood, model)

training_iter = 500
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_X_scaled)
    loss = -mll(output, train_Y_scaled.squeeze(-1))
    loss.backward()
    print(f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}")
    optimizer.step()


def evaluate_and_plot(
    model, likelihood, X, y_true, y_mean, y_std, data_type="Test", figsize=(8, 5)
):
    """
    Evaluate GP model and create prediction plot with uncertainty.

    Args:
        model: Trained GP model
        likelihood: Trained GP likelihood
        X: Input features (torch tensor)
        y_true: True target values (torch tensor)
        y_mean: Mean of training targets for un-scaling
        y_std: Std of training targets for un-scaling
        data_type: String to identify the data type in plot labels ("Train" or "Test")
        figsize: Figure size tuple
    """
    # ===== Predict with uncertainty =====
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X))
        mean_scaled = observed_pred.mean.cpu().numpy()
        var_scaled = observed_pred.variance.cpu().numpy()

    # Un-standardize
    mean = mean_scaled * y_std.numpy() + y_mean.numpy()
    var = var_scaled * (y_std.numpy() ** 2)

    y_true = y_true.squeeze(-1).numpy()

    # ===== Metrics =====
    rmse = np.sqrt(np.mean((mean - y_true) ** 2))
    mae = np.mean(np.abs(mean - y_true))

    print(f"{data_type} RMSE: {rmse:.2f} %BF")
    print(f"{data_type} MAE : {mae:.2f} %BF")

    # ===== Visualization: predicted vs. true with 95% CI =====
    order = np.argsort(y_true)  # sort for a nice ribbon plot
    yt = y_true[order]
    ym = mean[order]
    ys = np.sqrt(var[order])  # Convert variance to standard deviation
    lower = ym - 1.96 * ys
    upper = ym + 1.96 * ys

    plt.figure(figsize=figsize)
    plt.plot(yt, label="True (sorted)")
    plt.plot(ym, label="Predicted mean")
    plt.fill_between(range(len(ym)), lower, upper, alpha=0.2, label="95% CI")
    plt.xlabel(f"{data_type} samples (sorted by true %BF)")
    plt.ylabel("Body Fat %")
    plt.title(f"Body Fat % — GP predictions with uncertainty ({data_type})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rmse, mae


# ===== Evaluate on test data =====
test_rmse, test_mae = evaluate_and_plot(
    model, likelihood, test_X_scaled, test_Y, y_mean, y_std, "Test"
)

# ===== Evaluate on training data =====
train_rmse, train_mae = evaluate_and_plot(
    model, likelihood, train_X_scaled, train_Y, y_mean, y_std, "Train"
)
