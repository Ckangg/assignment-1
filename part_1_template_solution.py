# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
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

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        answer = {}

        # Enter your code and fill the `answer` dictionary

        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = Xtrain.max()
        answer["max_Xtest"] = Xtest.max()
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        clf = DecisionTreeClassifier(random_state=42)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        results=train_simple_classifier_with_cv(Xtrain,ytrain,clf,cv)
        pc1={}
        pc1['mean_fit_time']=results['fit_time'].mean()
        pc1['std_fit_time']=results['fit_time'].std()
        pc1['mean_accuracy']=results['test_score'].mean()
        pc1['std_accuracy']=results['test_score'].std()
        answer = {}
        answer["clf"] =clf  # the estimator (classifier instance)
        answer["cv"] = cv  # the cross validator instance
        # the dictionary with the scores  (a dictionary with
        # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
        answer["scores"] = pc1
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary

        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'
        clf = DecisionTreeClassifier(random_state=42)

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        results=train_simple_classifier_with_cv(Xtrain,ytrain,clf,cv)
        pd1={}
        pd1['mean_fit_time']=results['fit_time'].mean()
        pd1['std_fit_time']=results['fit_time'].std()
        pd1['mean_accuracy']=results['test_score'].mean()
        pd1['std_accuracy']=results['test_score'].std()
        answer = {}
        answer["clf"] = clf
        answer["cv"] = cv
        answer["scores"] = pd1
        answer["explain_kfold_vs_shuffle_split"] = "advantages:It allows precise control over the size of each iteration's training and test sets, unlike k-fold cross-validation, which divides the dataset into k parts of equal size.By randomly sampling to create training and test sets in each iteration, it can reduce the variance of evaluation metrics, making the results more stable.Particularly suited for very large datasets, where complete k-fold cross-validation would be too costly or time-consuming. disadvantages:Due to random sampling, there might be an overlap between training and test sets, especially across multiple iterations. This could lead to evaluation metrics being overly optimistic.If some categories within the dataset are very rare, random sampling might not guarantee that each split represents the whole dataset's structure well, especially with fewer iterations."
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`
        clf = DecisionTreeClassifier(random_state=42)
        ks = [2, 5, 8, 16]  # Different values of k
        pe1={}
        for k in ks:
            cv = ShuffleSplit(n_splits=k, test_size=0.2, random_state=42)
            scores = cross_validate(clf, Xtrain, ytrain, cv=cv, scoring='accuracy', return_train_score=False)
    
            mean_accuracy = np.mean(scores['test_score'])
            std_accuracy = np.std(scores['test_score'])
            mean_fit_time=np.mean(scores['fit_time'])
            std_fit_time=np.std(scores['fit_time'])
            pe1[k] = {'Mean accuracy': mean_accuracy, 'Standard deviation': std_accuracy, 'Mean fit time': mean_fit_time, 'Standard deviation of fit time': std_fit_time}
            
    
        for k, metrics in pe1.items():
            print(f"k={k}: Mean accuracy = {metrics['Mean accuracy']:.4f}, Standard deviation = {metrics['Standard deviation']:.4f}, Mean fit time = {metrics['Mean fit time']:.4f}, Standard deviation of fit time = {metrics['Standard deviation of fit time']:.4f}")
        answer = {}
        answer["clf"] = clf
        answer["cv"] = cv
        answer["scores"] = pe1
        

        # Enter your code, construct the `answer` dictionary, and return it.

        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """

        answer = {}
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        # Decision Tree Classifier
        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_scores = cross_validate(dt_clf, Xtrain, ytrain, cv=cv, scoring='accuracy', return_train_score=False)
        
        # Random Forest Classifier
        rf_clf = RandomForestClassifier(random_state=42)
        rf_scores = cross_validate(rf_clf, Xtrain, ytrain, cv=cv, scoring='accuracy', return_train_score=False)
        
        # Calculate mean accuracy and standard deviation for both models
        
        dt_mean_accuracy = np.mean(dt_scores['test_score'])
        dt_std_accuracy = np.std(dt_scores['test_score'])
        dt_mean_fit_time=np.mean(dt_scores['fit_time'])
        dt_std_fit_time=np.std(dt_scores['fit_time'])
        
        rf_mean_accuracy = np.mean(rf_scores['test_score'])
        rf_std_accuracy = np.std(rf_scores['test_score'])
        rf_mean_fit_time=np.mean(rf_scores['fit_time'])
        rf_std_fit_time=np.std(rf_scores['fit_time'])
        
        
        # Determine which model has the highest accuracy on average and the lowest variance
        highest_accuracy_model = "Random Forest" if rf_mean_accuracy > dt_mean_accuracy else "Decision Tree"
        lowest_variance_model = "Random Forest" if rf_std_accuracy < dt_std_accuracy else "Decision Tree"
        fastest_model="Random Forest" if rf_mean_fit_time < dt_mean_fit_time else "Decision Tree"
        
        highest_accuracy_model, lowest_variance_model,fastest_model
                # Enter your code, construct the `answer` dictionary, and return it.
        scores_RF = {
            "mean_fit_time": np.mean(rf_scores['fit_time']),
            "std_fit_time": np.std(rf_scores['fit_time']),
            "mean_accuracy": np.mean(rf_scores['test_score']),
            "std_accuracy": np.std(rf_scores['test_score'])
        }
        
        scores_DT = {
            "mean_fit_time": np.mean(dt_scores['fit_time']),
            "std_fit_time": np.std(dt_scores['fit_time']),
            "mean_accuracy": np.mean(dt_scores['test_score']),
            "std_accuracy": np.std(dt_scores['test_score'])
        }
        answer = {
            "clf_RF": rf_clf,
            "clf_DT": dt_clf,
            "cv": cv,
            "scores_RF": scores_RF,
            "scores_DT": scores_DT,
            "model_highest_accuracy": model_highest_accuracy,
            "model_lowest_variance": model_lowest_variance,
            "model_fastest": model_fastest
        }
        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float)
            "model_fastest" (float)
        """

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        answer = {}
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Base estimator
        rf = RandomForestClassifier(random_state=42)
        
        # GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_79, y_train_79)
        
        # Best estimator
        best_rf = grid_search.best_estimator_
        
        # Fit the original and best estimator to compute confusion matrices
        rf.fit(X_train_79, y_train_79)  # Original estimator
        y_pred_train_orig = rf.predict(X_train_79)
        y_pred_test_orig = rf.predict(X_test_79)
        
        # Best estimator predictions
        y_pred_train_best = best_rf.predict(X_train_79)
        y_pred_test_best = best_rf.predict(X_test_79)
        
        # Confusion matrices
        confusion_matrix_train_orig = confusion_matrix(y_train_79, y_pred_train_orig)
        confusion_matrix_train_best = confusion_matrix(y_train_79, y_pred_train_best)
        confusion_matrix_test_orig = confusion_matrix(y_test_79, y_pred_test_orig)
        confusion_matrix_test_best = confusion_matrix(y_test_79, y_pred_test_best)
        
        # Accuracy scores
        accuracy_orig_full_training = accuracy_score(y_train_79, y_pred_train_orig)
        accuracy_best_full_training = accuracy_score(y_train_79, y_pred_train_best)
        accuracy_orig_full_testing = accuracy_score(y_test_79, y_pred_test_orig)
        accuracy_best_full_testing = accuracy_score(y_test_79, y_pred_test_best)
        
        # Answer dictionary
        answer = {
            "clf": RandomForestClassifier(random_state=42),  # Base classifier instance
            "default_parameters": RandomForestClassifier().get_params(),  # Default parameters of the classifier
            "best_estimator": best_rf,  # Classifier with the best parameters found
            "grid_search": grid_search,  # GridSearchCV instance
            "mean_accuracy_cv": grid_search.best_score_,  # Mean CV accuracy
            "confusion_matrix_train_orig": confusion_matrix_train_orig,  # Confusion matrix of training data with initial estimator
            "confusion_matrix_train_best": confusion_matrix_train_best,  # Confusion matrix of training data with best estimator
            "confusion_matrix_test_orig": confusion_matrix_test_orig,  # Confusion matrix of test data with initial estimator
            "confusion_matrix_test_best": confusion_matrix_test_best,  # Confusion matrix of test data with best estimator
            "accuracy_orig_full_training": accuracy_orig_full_training,  # Accuracy of training data with initial estimator
            "accuracy_best_full_training": accuracy_best_full_training,  # Accuracy of training data with best estimator
            "accuracy_orig_full_testing": accuracy_orig_full_testing,  # Accuracy of test data with initial estimator
            "accuracy_best_full_testing": accuracy_best_full_testing  # Accuracy of test data with best estimator
        }

        # Enter your code, construct the `answer` dictionary, and return it.

        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """

        return answer
