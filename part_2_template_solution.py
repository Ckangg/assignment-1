# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        answer = {}
        nb_classes_train = len(np.unique(y_train))
        nb_classes_test = len(np.unique(y_test))
        class_count_train = Counter(y_train)
        class_count_test = Counter(y_test)
        
        # Calculate lengths of the datasets and their labels
        length_Xtrain = X_train.shape[0]
        length_Xtest = X_test.shape[0]
        length_ytrain = len(y_train)
        length_ytest = len(y_test)
        
        # Find the maximum values in the training and testing datasets
        max_Xtrain = X_train.max()
        max_Xtest = X_test.max()
        
        # Fill the answer dictionary
        answer = {
            "nb_classes_train": nb_classes_train,
            "nb_classes_test": nb_classes_test,
            "class_count_train": class_count_train,
            "class_count_test": class_count_test,
            "length_Xtrain": length_Xtrain,
            "length_Xtest": length_Xtest,
            "length_ytrain": length_ytrain,
            "length_ytest": length_ytest,
            "max_Xtrain": max_Xtrain,
            "max_Xtest": max_Xtest
        }
        
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        answer = {}
       
        pb2={}
        sizes = [(1000, 200), (5000, 1000), (10000, 2000)]
        X_train, y_train, X_test, y_test = prepare_data()
        if scale(X_train) and scale(X_test):
            print("Scaling successful")
        else:
            print("Scaling failed")
        for ntrain, ntest in sizes:
                    Xtrain, ytrain = X[:ntrain], y[:ntrain]
                    Xtest, ytest = X[ntrain:ntrain+ntest], y[ntrain:ntrain+ntest]
                    clf = DecisionTreeClassifier(random_state=42)
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    results=train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain,clf=clf,cv=cv)
                    pb2 = {
                            'mean_fit_time': results['fit_time'].mean(),
                            'std_fit_time': results['fit_time'].std(),
                            'mean_accuracy': results['test_score'].mean(),
                            'std_accuracy': results['test_score'].std(),
                            "clf": clf,  
                            "cv": cv
                        }
                    clf = DecisionTreeClassifier(random_state=42)
                    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                    results=train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain,clf=clf,cv=cv)
                    pb3 = {
                            'mean_fit_time': results['fit_time'].mean(),
                            'std_fit_time': results['fit_time'].std(),
                            'mean_accuracy': results['test_score'].mean(),
                            'std_accuracy': results['test_score'].std(),
                            "clf": clf,  
                            "cv": cv
                        }
                    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                    clf = LogisticRegression(max_iter=300, multi_class='ovr', solver='lbfgs')
                    clf.fit(Xtrain, ytrain)
                    results=train_simple_classifier_with_cv(Xtrain=Xtrain,ytrain=ytrain,clf=clf,cv=cv)
                    pb4 = {
                            'mean_fit_time': results['fit_time'].mean(),
                            'std_fit_time': results['fit_time'].std(),
                            'mean_accuracy': results['test_score'].mean(),
                            'std_accuracy': results['test_score'].std(),
                            "clf": clf,  
                            "cv": cv
                        }
                    class_count_train = list(Counter(ytrain).values())
                    class_count_test = list(Counter(ytest).values())
                    key = f"train_{ntrain}_test_{ntest}"
                    answer[key] = {
                    "partC": pb2,
                    "partD": pb3,
                    "partF": pb4,
                    "ntrain": ntrain,
                    "ntest": ntest,
                    "class_count_train": class_count_train,
                    "class_count_test": class_count_test,
                    }
        answer=answer
        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """

        return answer
