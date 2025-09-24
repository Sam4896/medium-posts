from sklearn.model_selection import train_test_split
import torch
from data import load_data, order_data

from model import create_botorch_model
from train import fit_botorch_model
from evaluate import evaluate_and_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """
    Main function to run a single train/test split evaluation of the GP model.
    """
    # Load data
    X, y = load_data(remove_outliers_flag=False)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=20
    )

    X_train, y_train = order_data(X_train, y_train)
    X_test, y_test = order_data(X_test, y_test)

    # Convert to torch tensors
    train_X: torch.Tensor = torch.tensor(X_train, dtype=torch.double).to(device)
    train_Y: torch.Tensor = torch.tensor(y_train, dtype=torch.double).to(device)
    test_X: torch.Tensor = torch.tensor(X_test, dtype=torch.double).to(device)
    test_Y: torch.Tensor = torch.tensor(y_test, dtype=torch.double).to(device)

    # Initialize and train model
    print("--- Training model ---")
    model = create_botorch_model(train_X, train_Y).to(device, dtype=torch.double)
    model = fit_botorch_model(model)

    # Evaluate on test data
    print("\n--- Evaluating on test data ---")
    evaluate_and_plot(model, test_X, test_Y, "Test", likelihood=model.likelihood)

    # Evaluate on training data
    print("\n--- Evaluating on training data ---")
    evaluate_and_plot(model, train_X, train_Y, "Train", likelihood=model.likelihood)


if __name__ == "__main__":
    main()
