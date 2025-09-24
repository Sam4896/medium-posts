import torch
from gpytorch.likelihoods import StudentTLikelihood
from gpytorch.means import LinearMean
from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import PolynomialKernel


def create_botorch_model(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    """
    Creates a BoTorch SingleTaskGP model with input and outcome transforms.
    """
    # Create input and outcome transforms
    input_transform = Normalize(d=train_x.shape[-1])
    outcome_transform = Standardize(m=train_y.shape[-1])

    mean_module = LinearMean(input_size=train_x.shape[1])
    covar_module = PolynomialKernel(power=2)

    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        mean_module=mean_module,
        covar_module=covar_module,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )

    return model


def create_studentT_botorch_model(
    train_x: torch.Tensor, train_y: torch.Tensor
) -> SingleTaskVariationalGP:
    """
    Creates a BoTorch SingleTaskVariationalGP model with a StudentT likelihood.
    """
    # Create input and outcome transforms
    input_transform = Normalize(d=train_x.shape[-1])
    outcome_transform = Standardize(m=train_y.shape[-1])

    noise_prior = LogNormalPrior(loc=-6.0, scale=0.1)
    MIN_INFERRED_NOISE_LEVEL = (1e-4,)
    likelihood = StudentTLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            transform=None,
            initial_value=noise_prior.mode,
        ),
    )

    mean_module = LinearMean(input_size=train_x.shape[1]).to(
        train_x.device, dtype=train_x.dtype
    )
    covar_module = PolynomialKernel(power=2).to(train_x.device, dtype=train_x.dtype)

    model = SingleTaskVariationalGP(
        train_X=train_x,
        train_Y=train_y,
        likelihood=likelihood,
        mean_module=mean_module,
        covar_module=covar_module,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        # inducing_points=train_x.shape[-2],
    )

    return model
