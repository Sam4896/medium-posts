import pandas as pd
import numpy as np
import kagglehub
import os
from typing import Tuple, List, Optional


def load_data(
    remove_outliers_flag: bool = True, age_range: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the body fat dataset."""
    path: str = kagglehub.dataset_download("fedesoriano/body-fat-prediction-dataset")
    csv_path: str = os.path.join(path, "bodyfat.csv")
    df: pd.DataFrame = pd.read_csv(csv_path)

    target_col: str = "BodyFat"
    # target_col: str = "Density"
    candidate_features: List[str] = [
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
    features: List[str] = [c for c in candidate_features if c in df.columns]

    df = df.dropna(subset=[target_col] + features).reset_index(drop=True)

    X: np.ndarray = df[features].values.astype(np.float64)
    y: np.ndarray = df[[target_col]].values.astype(np.float64)

    if remove_outliers_flag:
        X, y = remove_outliers(X, y)

    if age_range is not None:
        X, y = filter_data_agewise(X, y, age_range)
    return X, y


def filter_data_agewise(
    X: np.ndarray, y: np.ndarray, age_range: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Filters the data by age."""
    mask = (X[:, 0] >= age_range[0]) & (X[:, 0] <= age_range[1])
    return X[mask], y[mask]


def remove_outliers(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Removes outliers from the data based on the target variable using the IQR method."""
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (y > lower_bound) & (y < upper_bound)
    mask = mask.squeeze()

    num_outliers = len(y) - np.sum(mask)
    if num_outliers > 0:
        print(f"Removed {num_outliers} outliers.")

    return X[mask], y[mask]


def order_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orders the data by the target variable."""
    order = np.argsort(y, axis=0).squeeze()
    return X[order], y[order]
