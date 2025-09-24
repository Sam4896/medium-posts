import torch
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from botorch.fit import fit_gpytorch_mll
from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskVariationalGP


def fit_botorch_model(model: GPyTorchModel) -> GPyTorchModel:
    """
    Fits a BoTorch model using fit_gpytorch_mll.
    """
    model.train()
    if isinstance(model, SingleTaskVariationalGP):
        model.train()
        mll = VariationalELBO(
            model.likelihood, model.model, num_data=model.model.train_targets.shape[-1]
        )
        fit_gpytorch_mll(mll)
    else:
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    model.eval()
    return model


def train_model_custom_optimizer(
    model: GPyTorchModel,
    training_iter: int = 500,
    lr: float = 0.1,
    verbose: bool = True,
) -> GPyTorchModel:
    """
    Trains the GP model by optimizing the marginal log likelihood with a custom optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if isinstance(model, SingleTaskVariationalGP):
        model.model.train()
        model.likelihood.train()
        mll = VariationalELBO(
            model.likelihood, model.model, num_data=model.model.train_targets.shape[-1]
        )
        train_inputs = model.model.train_inputs[0]
        train_targets = model.model.train_targets
    else:
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        train_inputs = model.train_inputs[0]
        train_targets = model.train_targets

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_inputs)
        loss = -mll(output, train_targets)
        loss.backward()
        if verbose and (i + 1) % 50 == 0:
            print(f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}")
        optimizer.step()

    model.eval()
    return model
