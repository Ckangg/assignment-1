import pickle
import numpy as np

from collections import Counter
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from typing import Type, Dict
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_validate, StratifiedKFold
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import (
    cross_validate,
    KFold,
)


def load_mnist_dataset(
    nb_samples=None,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """
    Load the MNIST dataset.

    nb_samples: number of samples to save. Useful for code testing.
    The homework requires you to use the full dataset.

    Returns:
        X, y
        #X_train, y_train, X_test, y_test
    """

    try:
        # Are the datasets already loaded?
        print("... Is MNIST dataset local?")
        X: NDArray[np.floating] = np.load("mnist_X.npy")
        y: NDArray[np.int32] = np.load("mnist_y.npy", allow_pickle=True)
    except Exception as e:
        # Download the datasets
        print(f"load_mnist_dataset, exception {e}, Download file")
        X, y = datasets.fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False
        )
        X = X.astype(float)
        y = y.astype(int)

    y = y.astype(np.int32)
    X: NDArray[np.floating] = X
    y: NDArray[np.int32] = y

    if nb_samples is not None and nb_samples < X.shape[0]:
        X = X[0:nb_samples, :]
        y = y[0:nb_samples]

    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    np.save("mnist_X.npy", X)
    np.save("mnist_y.npy", y)
    return X, y


def prepare_data(num_train: int = 60000, num_test: int = 10000, normalize: bool = True):
    """
    Prepare the data.
    Parameters:
        X: A data matrix
        frac_train: Fraction of the data used for training in [0,1]
    Returns:
    Prepared data matrix
    Side effect: update of self.Xtrain, self.ytrain, self.Xtest, self.ytest
    """
    # Check in case the data is already on the computer.
    X, y = load_mnist_dataset()

    # won't work well unless X is greater or equal to zero
    if normalize:
        X = X / X.max()

    y = y.astype(np.int32)
    Xtrain, Xtest = X[:num_train], X[num_train : num_train + num_test]
    ytrain, ytest = y[:num_train], y[num_train : num_train + num_test]
    return Xtrain, ytrain, Xtest, ytest


def filter_out_7_9s(X: NDArray[np.floating], y: NDArray[np.int32]):
    """
    Filter the dataset to include only the digits 7 and 9.
    Parameters:
        X: Data matrix
        y: Labels
    Returns:
        Filtered data matrix and labels
    Notes:
        np.int32 is a type with a range based on 32-bit ints
        np.int has no bound; it can hold arbitrarily long numbers
    """
    seven_nine_idx = (y == 7) | (y == 9)
    X_binary = X[seven_nine_idx, :]
    y_binary = y[seven_nine_idx]
    return X_binary, y_binary


def train_simple_classifier_with_cv(
    *,
    Xtrain: NDArray[np.floating],
    ytrain: NDArray[np.int32],
    clf: BaseEstimator,
    cv: KFold = KFold,
):
    """
    Train a simple classifier using k-vold cross-validation.

    Parameters:
        - X: Features dataset.
        - y: Labels.
        - cv_class: The cross-validation class to use.
        - estimator_class: The training classifier class to use.
        - n_splits: Number of splits for cross-validation.
        - print_results: Whether to print the results.

    Returns:
        - A dictionary with mean and std of accuracy and fit time.
    """
    scores = cross_validate(clf, Xtrain, ytrain, cv=cv)
    return scores


def print_cv_result_dict(cv_dict: Dict):
    for key, array in cv_dict.items():
        print(f"mean_{key}: {array.mean()}, std_{key}: {array.std()}")


def starter_code():
    try:
        Xtrain, ytrain, Xtest, ytest = prepare_data()
        Xtrain, ytrain = filter_out_7_9s(Xtrain, ytrain)
        Xtest, ytest = filter_out_7_9s(Xtest, ytest)
        out_dict = train_simple_classifier_with_cv(
            Xtrain=Xtrain, ytrain=ytrain, clf=DecisionTreeClassifier(),
            cv=KFold(n_splits=3)
        )
        print_cv_result_dict(out_dict)
        return 0
    except Exception:
        return -1  # Error


def save_dict(filenm, dct):
    with open(filenm, "wb") as file:
        pickle.dump(dct, file)


# Loading from a pickle file
def load_dict(filenm, dct):
    with open(filenm, "rb") as file:
        loaded_data = pickle.load(file)
        return loaded_data
    raise "load_dict:: Error loading data"


if __name__ == "__main__":
    starter_code()

def scale(X):
    """
    Scale the elements of the data matrix to be between 0 and 1.
    
    Parameters:
        X (np.ndarray): The data matrix to scale.
        
    Returns:
        bool: True if scaling was successful, False otherwise.
    """
    try:
        X_max = X.max()
        if X_max > 0:
            X /= X_max
        return True
    except Exception as e:
        print(f"Error while scaling: {e}")
        return False
