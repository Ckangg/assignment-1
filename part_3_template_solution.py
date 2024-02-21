import numpy as np
from numpy.typing import NDArray
from typing import Any

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary
        X,y =load_mnist_dataset()
        if scale(X):
             print("Scaling successful")
        else:
             print("Scaling failed")
        X_train, y_train, X_test, y_test = prepare_data()
         
        if scale(X_train) and scale(X_test):
             print("Scaling successful")
        else:
             print("Scaling failed")
         
        ntrains = [1000, 5000, 10000]
        ntests = [200, 1000, 2000]
        ks = [1, 2, 3, 4, 5]
         
        answer = {}
         
        for ntrain, ntest in zip(ntrains, ntests):
             # Assuming X and y are predefined datasets
             Xtrain, Xtest = X[:ntrain], X[ntrain:ntrain+ntest]
             ytrain, ytest = y[:ntrain], y[ntrain:ntrain+ntest]
             
             model = LogisticRegression(max_iter=300, multi_class='multinomial', solver='lbfgs')
             model.fit(Xtrain, ytrain)
             
             train_scores = [top_k_accuracy_score(ytrain, model.predict_proba(Xtrain), k=k) for k in ks]
             test_scores = [top_k_accuracy_score(ytest, model.predict_proba(Xtest), k=k) for k in ks]
             
             plt.figure(figsize=(8, 5))
             plt.plot(ks, train_scores, label='Training', marker='o')
             plt.plot(ks, test_scores, label='Testing', marker='o')
             plt.xlabel('k')
             plt.ylabel('Top-k Accuracy')
             plt.title(f'Top-k Accuracy vs. k (ntrain={ntrain}, ntest={ntest})')
             plt.legend()
             plt.grid(True)
             plt.show()
             
             for k, score_train, score_test in zip(ks, train_scores, test_scores):
                 answer[k] = {"score_train": score_train, "score_test": score_test}
             
             answer["clf"] = "Logistic Regression"
             answer["clf2"]= model
             answer["plot_k_vs_score_train"] = list(zip(ks, train_scores))
             answer["plot_k_vs_score_test"] = list(zip(ks, test_scores))
             
             # Additional analysis on rate of accuracy change for testing data
             # This is a placeholder for detailed analysis
             answer["text_rate_accuracy_change"] = "with the K becomes larger, the test accuracy rate becomes higher."
             
             # Comments on the usefulness of top-k accuracy metric
             answer["text_is_topk_useful_and_why"] = "Top-k accuracy is useful when there are multiple classifers, in this situation, the topk accuracy is more suitable to judge the ability of claasifier than accuracy.  " \
                                          
        

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        answer={}
        Xtrain, ytrain, Xtest, ytest=prepare_data()
        Xtrain, ytrain= filter_out_7_9s(Xtrain, ytrain)
        Xtest, ytest= filter_out_7_9s(Xtest, ytest)
        ytrain = ytrain.astype(int)
        ytrain[ytrain == 7] = 0
        ytrain[ytrain == 9] = 1
        ytest = ytest.astype(int)
        ytest[ytest == 7] = 0
        ytest[ytest == 9] = 1
        mask = (np.random.rand(len(ytrain)) < 0.1) | (ytrain == 0)
        Xtrain = Xtrain[mask]
        ytrain = ytrain[mask]
        mask = (np.random.rand(len(ytest)) < 0.1) | (ytest == 0)
        Xtest = Xtest[mask]
        ytest = ytest[mask]
        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = Xtrain.max()
        answer["max_Xtest"] = Xtest.max()

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
        answer = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        svc_classifier = SVC(random_state=42)
        scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'f1_score': make_scorer(f1_score, average='macro')}
        svc_cv_results = cross_validate(svc_classifier, Xtrain, ytrain, cv=cv, scoring=scoring)
        for metric_name, scores in svc_cv_results.items():
           print(f"{metric_name}: Mean={np.mean(scores)}, Std={np.std(scores)}")
        scores_summary = {metric: {"mean": np.mean(scores), "std": np.std(scores)} 
                  for metric, scores in svc_cv_results.items() if 'test_' in metric}
        is_precision_higher_than_recall = scores_summary['test_precision']['mean'] > scores_summary['test_recall']['mean']
        classifier = SVC(random_state=42)
        X2 = np.concatenate((Xtrain, Xtest), axis=0)
        y2 = np.concatenate((ytrain, ytest), axis=0)
        classifier.fit(X2, y2)
        y_pred = classifier.predict(X2)
        conf_mat = confusion_matrix(y2, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_mat, annot=True, fmt='g')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        answer = {
        "scores": {
        "mean_accuracy": scores_summary['test_accuracy']['mean'],
        "mean_recall": scores_summary['test_recall']['mean'],
        "mean_precision": scores_summary['test_precision']['mean'],
        "mean_f1": scores_summary['test_f1_score']['mean'],
        "std_accuracy": scores_summary['test_accuracy']['std'],
        "std_recall": scores_summary['test_recall']['std'],
        "std_precision": scores_summary['test_precision']['std'],
        "std_f1": scores_summary['test_f1_score']['std']
        },
        "cv": cv,
        "clf": svc_classifier,
        "is_precision_higher_than_recall": is_precision_higher_than_recall,
        "explain_is_precision_higher_than_recall": "the model is more likely to predict the positive class, because it is an imbalanced dataset, so if we don't change the weights of change the threshold, it will show the imbalanced result",
        "confusion_matrix": conf_mat  # Confusion matrix for the combined dataset
        }
        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        answer = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(ytrain), y=ytrain)
        class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
        print("Class weights:", class_weights_dict)
        svc_classifier = SVC(class_weight=class_weights_dict, random_state=42)
        scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'f1_score': make_scorer(f1_score, average='macro')}
        svc_cv_results = cross_validate(svc_classifier, Xtrain, ytrain, cv=cv, scoring=scoring)
        for metric_name, scores in svc_cv_results.items():
           print(f"{metric_name}: Mean={np.mean(scores)}, Std={np.std(scores)}")
        scores_summary = {metric: {"mean": np.mean(scores), "std": np.std(scores)} 
                  for metric, scores in svc_cv_results.items() if 'test_' in metric}
        is_precision_higher_than_recall = scores_summary['test_precision']['mean'] > scores_summary['test_recall']['mean']
       
        svc_classifier = SVC(class_weight=class_weights_dict, random_state=42)
        svc_classifier.fit(Xtrain, ytrain)
        y_pred = svc_classifier.predict(Xtrain)
        conf_mattrain = confusion_matrix(ytrain, y_pred)
        y_pred = svc_classifier.predict(Xtest)
        conf_mattest = confusion_matrix(ytest, y_pred)
        
        answer = {
        "scores": {
        "mean_accuracy": scores_summary['test_accuracy']['mean'],
        "mean_recall": scores_summary['test_recall']['mean'],
        "mean_precision": scores_summary['test_precision']['mean'],
        "mean_f1": scores_summary['test_f1_score']['mean'],
        "std_accuracy": scores_summary['test_accuracy']['std'],
        "std_recall": scores_summary['test_recall']['std'],
        "std_precision": scores_summary['test_precision']['std'],
        "std_f1": scores_summary['test_f1_score']['std']
        },
        "cv": cv,
        "clf": svc_classifier,
        "class_weights": class_weights,
        "explain_purpose_of_class_weights":"use class_weights it can help balance the dataset, even one label is smaller than another, the model can still distinguish it because it has a higher weight",
        "explain_performance_difference":"no",
        "is_precision_higher_than_recall": is_precision_higher_than_recall,
        "explain_is_precision_higher_than_recall": "use the classweight to reduce the influence of imbalanced data, it will be useful to improve the model",
        "confusion_matrix_train": conf_mattrain,
        "confusion_matrix_test": conf_mattest}

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
