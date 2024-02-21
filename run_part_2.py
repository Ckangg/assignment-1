#######################################################################
# from sklearn.calibration import LinearSVC
# from sklearn.base import BaseEstimator
# from sklearn.model_selection import KFold
# from sklearn.tree import DecisionTreeClassifier

import numpy as np
import utils as u
from numpy.typing import NDArray
from part_2_template_solution import Section2 as Part2

#######################################################################

# The argument `self` refers to a class, specifically class Part2 which is a shortcut for Section2

# Run this file from the command line or from VSCode
#   python run_part2.py

"""
A. Repeat part 1.B but make sure that your data matrix (and labels) consists of all 
   classes by also printing out the number of elements in each class y and print out 
   the number of classes. 
"""


def part_A(
    self,
    # Xtrain: NDArray[np.floating],
    # ytrain: NDArray[np.int32],
    # Xtest: NDArray[np.floating],
    # ytest: NDArray[np.int32],
):
    # Do not filter out data
    # Return a dictionary for the training set and another for the testing set: key is the class, value is the number of elements in the class
    # The function should
    # Test: all values > 0 and sum of values = number of elements in the set
    # Test: Use fake data with wrong properties

    # answer = self.partA(Xtrain, ytrain, Xtest, ytest)
    answer, Xtrain, ytrain, Xtest, ytest = self.partA()
    return answer, Xtrain, ytrain, Xtest, ytest


"""
B. Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. Use the Logistic Regression for part F with 300 iterations. Explain how multi-class logistic 
regression works (inherent, one-vs-one, one-vs-the-rest, etc.). 
Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000. 
Comment on the results. Is the accuracy higher for the training or testing set?
What is the scores as a function of ntrain. 

Given X, y from mnist, use: 
Xtrain = X[0:ntrain, :]
ytrain = y[0:ntrain]
Xtest = X[ntrain:ntrain+test]
ytest = y[ntrain:ntrain+test]
"""


def part_B(
    self,
    X: NDArray[np.floating],
    y: NDArray[np.int32],
    Xtest: NDArray[np.floating],
    ytest: NDArray[np.int32],
    ntrain_list: list = [800],
    ntest_list: list = [64],
):
    answer = self.partB(X, y, Xtest, ytest, ntrain_list, ntest_list)
    return answer


# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """

    # Attention: the seed should never be changed. If it is, automatic grading
    # of the assignment could very well fail, and you'd lose points.
    # Make sure that all sklearn functions you use that require a seed have this
    # seed specified in the argument list, namely: `random_state=self.seed` if
    # you are inside the solution class.
    part2 = Part2(seed=42, frac_train=0.2)

    # X and Y are Mnist datasets
    # part_A(part2, X, y, Xtest, ytest)
    # Return values properly filtered, tec.
    answer = {}
    answer["2A"], Xtrain, ytrain, Xtest, ytest = part_A(part2)
    ntrain_list = [16000, 8000, 4000, 2000]
    ntest_list = [i // 4 for i in ntrain_list]
    answer["2B"] = part_B(part2, Xtrain, ytrain, Xtest, ytest, ntrain_list, ntest_list)
    u.save_dict("section2.pkl", answer)
