import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.likelihoods import Likelihood

FONT_SIZE = 12


def evaluate_and_plot(
    model: GPyTorchModel,
    X: torch.Tensor,
    y_true: torch.Tensor,
    data_type: str = "Test",
    figsize: Tuple[int, int] = (8, 5),
    plot: bool = True,
    likelihood: Optional[Likelihood] = None,
) -> Tuple[float, float]:
    """
    Evaluate GP model and create prediction plot with uncertainty.
    """
    # ===== Predict with uncertainty =====
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model.posterior(X)
        mean: np.ndarray = observed_pred.mean.squeeze(-1).cpu().numpy()
        lower, upper = observed_pred.confidence_region()
        upper = upper.squeeze(-1).cpu().numpy()
        lower = lower.squeeze(-1).cpu().numpy()

    y_true_numpy: np.ndarray = y_true.squeeze(-1).cpu().numpy()

    # ===== Metrics =====
    rmse: float = float(np.sqrt(np.mean((mean - y_true_numpy) ** 2)))
    mae: float = float(np.mean(np.abs(mean - y_true_numpy)))

    print(f"{data_type} RMSE: {rmse:.2f} %BF")
    print(f"{data_type} MAE : {mae:.2f} %BF")

    # ===== Visualization: predicted vs. true with 95% CI =====
    if plot:
        plt.rcParams.update({"font.size": FONT_SIZE})
        plt.figure(figsize=figsize)
        # Parity plot
        plt.errorbar(
            y_true_numpy,
            mean,
            yerr=np.stack([mean - lower, upper - mean]),
            fmt="o",
            ecolor="lightgray",
            elinewidth=3,
            capsize=0,
            label="Predicted mean with 95% CI",
        )
        # y=x line
        lims = [
            np.min([0, 40]),  # min of both axes
            np.max([0, 40]),  # max of both axes
        ]
        plt.plot(lims, lims, "k-", alpha=0.75, zorder=0, label="Ideal")
        plt.xlim(lims)
        plt.ylim(lims)

        plt.xlabel("True Body Fat %", fontsize=FONT_SIZE)
        plt.ylabel("Predicted Body Fat %", fontsize=FONT_SIZE)
        if likelihood is not None:
            if isinstance(likelihood, gpytorch.likelihoods.StudentTLikelihood):
                title_suffix = "Student-T"
            else:
                title_suffix = "Gaussian"
        plt.title(
            f"Parity Plot — GP predictions with uncertainty with {title_suffix} likelihood",
            fontsize=FONT_SIZE,
        )
        plt.legend(fontsize=FONT_SIZE)
        plt.tight_layout()
        plt.show()

    return rmse, mae
