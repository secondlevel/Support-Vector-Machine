from tkinter import font
from tokenize import single_quoted
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import axes


def draw_heatmap(gamma_list, C_list, data):

    xlabel = gamma_list
    ylabel = C_list
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(ylabel)))
    ax.set_yticklabels(ylabel)
    ax.set_xticks(range(len(xlabel)))
    ax.set_xticklabels(xlabel)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

    data = np.array(data)

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            text = ax.text(y, x, data[x, y],
                           ha="center", va="center", color="w")

    im = ax.imshow(data, cmap=plt.cm.coolwarm)
    plt.colorbar(im)

    plt.title("Hyperparameter GridSearch")
    plt.xlabel("Gamma Parameter")
    plt.ylabel("C Parameter")

    plt.savefig("Hyperparameter_GridSearch.png")
    plt.close()

    return None


# ## Load data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# 550 data with 300 features
print(x_train.shape)

# It's a binary classification problem
print(np.unique(y_train))

# ## Question 1
# K-fold data partition: Implement the K-fold cross-validation function. Your function should take K as an argument and return a list of lists (len(list) should equal to K), which contains K elements. Each element is a list contains two parts, the first part contains the index of all training folds, e.g. Fold 2 to Fold 5 in split 1. The second part contains the index of validation fold, e.g. Fold 1 in  split 1


def cross_validation(x_train, y_train, k=5):
    kfold_data = []
    single_fold_num = len(x_train)//k
    shuffle_index = np.arange(len(x_train))
    np.random.seed(14)
    np.random.shuffle(shuffle_index)
    for cur_k in range(0, len(x_train), single_fold_num):
        validation_fold = shuffle_index[cur_k:cur_k+single_fold_num]
        training_fold = np.array(
            [x for x in shuffle_index if x not in validation_fold])
        kfold_data.append([training_fold, validation_fold])

    if (len(x_train) % k) > 0:
        validation_fold = shuffle_index[cur_k:len(x_train)]
        training_fold = np.array(
            [x for x in shuffle_index if x not in validation_fold])
        kfold_data.append([training_fold, validation_fold])

    return kfold_data


kfold_data = cross_validation(x_train, y_train, k=10)
assert len(kfold_data) == 10  # should contain 10 fold of data
# each element should contain train fold and validation fold
assert len(kfold_data[0]) == 2
# The number of data in each validation fold should equal to training data divieded by K
assert kfold_data[0][1].shape[0] == 55

# ## example
X = np.arange(20)
kf = KFold(n_splits=5, shuffle=True)
kfold_data = []
for i, (train_index, val_index) in enumerate(kf.split(X)):
    print("Split: %s, Training index: %s, Validation index: %s" %
          (i+1, train_index, val_index))
    kfold_data.append([train_index, val_index])

assert len(kfold_data) == 5  # should contain 5 fold of data
# each element should contains index of training fold and validation fold
assert len(kfold_data[0]) == 2
# The number of data in each validation fold should equal to training data divieded by K
assert kfold_data[0][1].shape[0] == 4

# ## Question 2
# Using sklearn.svm.SVC to train a classifier on the provided train set and conduct the grid search of “C”, “kernel” and “gamma” to find the best parameters by cross-validation.

kfold_data = cross_validation(x_train, y_train, k=5)
# clf = SVC(C=1.0, kernel='rbf', gamma=0.01)

maximum_score = -1
best_parameters = []
picture_data = []

for kernel in ['rbf']:
    for C in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        validate_acc = []
        for gmma in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            average_score = []
            for idx, fold_data in enumerate(kfold_data):
                clf = SVC(C=C, kernel=kernel, gamma=gmma)
                clf.fit(x_train[fold_data[0]], y_train[fold_data[0]])
                y_pred = clf.predict(x_train[fold_data[1]])
                score = accuracy_score(y_pred, y_train[fold_data[1]])
                average_score.append(score)
            score = sum(average_score)/len(average_score)
            validate_acc.append(round(score, 2))
            if score > maximum_score:
                maximum_score = score
                best_parameters = {"kernel": kernel, "gamma": gmma, "C": C}
        picture_data.append(validate_acc)

draw_heatmap([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
             [0.01, 0.1, 1, 10, 100, 1000, 10000], picture_data)
print(best_parameters)

# ## Question 3
# Plot the grid search results of your SVM. The x, y represents the hyperparameters of “gamma” and “C”, respectively. And the color represents the average score of validation folds
# You reults should be look like the reference image ![image](https://miro.medium.com/max/1296/1*wGWTup9r4cVytB5MOnsjdQ.png)

# ## Question 4
# Train your SVM model by the best parameters you found from question 2 on the whole training set and evaluate the performance on the test set.

best_model = SVC(C=best_parameters["C"], kernel=best_parameters["kernel"],
                 gamma=best_parameters["gamma"])

best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("Accuracy score: ", accuracy_score(y_pred, y_test))
