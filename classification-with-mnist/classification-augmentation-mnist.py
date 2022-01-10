import numpy as np
from sklearn.datasets import fetch_openml
#[]
#
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

#[]
# going to get the target data (labels) and the features
mnist.keys()

#[]
#
mnist['data'].shape

#[]
#
mnist['target'].shape

#[]
# taking a quick look at the features
import matplotlib as mpl
import matplotlib.pyplot as plt
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

digit_1 = mnist['data'][2]
plot_digit(digit_1)
plt.show()

#[]
#
digit_1_l = mnist['target'][2]
digit_1_l


#[]
# both the features and labels have been adequately shuffled
# to prevent any problems that may occur in the algorithms
X_train, X_test = mnist['data'][:60000],mnist['data'][60000:]
y_train, y_test = mnist['target'][:60000],mnist['target'][60000:]


#[]
# plotting some of the digits
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

plot_digits(X_train[:50])
plt.show()
#[]
#
from collections import Counter
#[]
# going to convert to int

y_test = y_test.astype(np.uint8)
y_train = y_train.astype(np.uint8)

#[]
#
Counter(Y_train)

#[]
np.random.seed(42)

#[]
# stochastic gradient descent natively supports classification of multiple classes
# since dataset isn't too big I'm not going to mess with tolerance, max number of iterations, nor the learning
# rate for the moment
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

#[]
#
sgd_clf.fit(X_train, y_train)
#[]
#

sgd_clf.predict(digit_1)

#[]
#
plot_digit(digit_1)
plt.show()

#[]
#
digit_1_l

#[]
#
sgd_clf.classes_
#[]
# we can see with what certainty 4 was chosen
sgd_clf.decision_function([digit_1])
#[]
# not a robust way of measuring a model in itself when dealing with classification
# however we won't have too many problems since there is variety in
# our data but just to explore the results
# cross_valpredict is better due to us getting a prediction with a model that hasn't previously seen the data
# here the problem also is that this scoring of accuracy doesn't take into account false positives nor false negatives
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sgd_clf, X_train, y_train, scoring= 'accuracy', cv=3)



#[]
# going to get predictions that we can use for a confusion matrix
# note: some algorithms don't offer direct predictions but rather prediction probability
from sklearn.model_selection import cross_val_predict
sgd_y_train_predictions = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method='predict')


#[]
#
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, sgd_y_train_predictions)

#[]
# going to take a brief aside to check out some other metrics
# can you unweighted values since their is a good share of each number in the dataset
# the labels aren't  really imbalanced
from sklearn.metrics import precision_score
precision = precision_score(y_train, sgd_y_train_predictions, average='macro')
precision
# our precision tells us that when classifying numbers apprx. 86% match the labels

#[]
#
from sklearn.metrics import recall_score
recall = recall_score(y_train, sgd_y_train_predictions, average='macro')
recall

#[]
# f1 score to take both precision and recall into account
from sklearn.metrics import f1_score
score_f1 = f1_score(y_train, sgd_y_train_predictions, average='macro')
score_f1

#[]
# looking to use decision functions to plot precision recall curve
sgd_y_train_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=3, method='decision_function')




#[]
# multiclass format not supported however these would work well with some binary classifiers
# from sklearn.metrics import precision_recall_curve

# sgd_precisions, sgd_recalls, sgd_thresholds = precision_recall_curve(y_train, sgd_y_train_scores)

# #[]
# #
# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
#     plt.legend(loc="center right", fontsize=16)
#     plt.xlabel("Threshold", fontsize=16)
#     plt.grid(True)
#     plt.axis([-50000, 50000, 0, 1])

# plot_precision_recall_vs_threshold(sgd_precisions, sgd_recalls, sgd_thresholds)
# plt.show()

#[]
#
# def plot_precision_vs_recall(precisions, recalls):
#     plt.plot(recalls, precisions, "b-", linewidth=2)
#     plt.xlabel("Recall", fontsize=16)
#     plt.ylabel("Precision", fontsize=16)
#     plt.axis([0, 1, 0, 1])
#     plt.grid(True)

# plot_precision_vs_recall(sgd_precisions, sgd_recalls)
# plt.show()
# #[]
# # knowing that there is a tradeoff between precision and recall a good question would be which to prioritize
#  from sklearn.metrics import roc_curve
#  sgd_fpr, sgd_tpr, sgd_thresholds = roc_curve(y_train,sgd_y_train_predictions)

#[]
# with values from 0-255 we can use a standardscaler and see if that can improve our model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
scores = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3 , scoring='accuracy')
scores

#[]
# forgot to do this on first analysis so maybe the confusion matrix will change
sgd_y_train_predictions_scaled = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3, method='predict')
#[]
# going to plot another confusion matrix but we are going to edit it a bit to be able to get more information from it
conf_mx = confusion_matrix(y_train, sgd_y_train_predictions_scaled)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

#[]
#
sum_of_row = conf_mx.sum(axis=1)
norm_conf_mx = conf_mx / sum_of_row

#[]
# the problem with our previous plot however is that it would be better to see relative differences in the matrix
np.fill_diagonal(norm_conf_mx,0)
norm_conf_mx

#[]
# another method for normalization of confusion matrix
conf_mx_test = confusion_matrix(y_train, sgd_y_train_predictions_scaled, normalize='true')
np.fill_diagonal(conf_mx_test,0)
plt.matshow(conf_mx_test, cmap=plt.cm.gray)
plt.show()
# so some of the problems we can see seem logical many 4s and 7s are being predicted when in reality they are 9s
# also vice versa with many 7s being classified as 9s
# 8s are considered as 5s

#[]
# so we are going to use some boolean indexing to investigate this issue a bit more
# i wantrows where the prediction says one thing and then the actual label is another
label4,label9 = 4,9
four_but_nine = X_train[(sgd_y_train_predictions_scaled == label4) & (y_train == label9)]
nine_but_nine = X_train[(sgd_y_train_predictions_scaled == label9) & (y_train == label9)]
nine_but_four = X_train[(sgd_y_train_predictions_scaled == label9) & (y_train == label4)]
four_but_four = X_train[(sgd_y_train_predictions_scaled == label4) & (y_train == label4)]

#{]
#

plt.figure(figsize=(10,10))
plt.subplot(221); plot_digits(nine_but_nine[:20], images_per_row=5)
plt.subplot(222); plot_digits(four_but_nine[:20], images_per_row=5)
plt.subplot(223); plot_digits(nine_but_four[:20], images_per_row=5)
plt.subplot(224); plot_digits(four_but_four[:20], images_per_row=5)
plt.show()
# 9 incorrectly classifed as 4 in upper right
# nines correctly classified as nines in upper left
# 4s incorrectly classified as 9s in lower left
# 4s correctly classified as 4s in lower right
# to try and improve on these classifications we could possibly look to load in some more images
# such as images that aren't 4s to help the model bette understand what doesn't make an image a 4
# and the same concept would apply for 9s where we would import in some images that look likes 9s but arent
# we could also preprocess these images a bit which is what we'll do next

#[]
# looking to shift a number in 4 different directions and then add to the training set to see how it goes
from scipy.ndimage.interpolation import shift
#[]
#
plot_digit(digit_1)
plt.show()
#[]
#
from scipy.ndimage.interpolation import shift
digit_1_augmented = shift(digit_1.reshape(28,28),[-9,0],cval=0)
plot_digit(digit_1_augmented)
plt.show()

#{]
#
def image_augment(image_pixels, x, y):
    reshaped_image = image_pixels.reshape(28,28)
    augmented_digit = shift(reshaped_image,[x,y])
    return augmented_digit.reshape(784,)

X_train_augmented = []
y_train_augmented = []
for row in range(len(X_train_scaled)):
    X_train_augmented.append(image_augment(X_train_scaled[row],x=1,y=0))
    X_train_augmented.append(image_augment(X_train_scaled[row],x=-1,y=0))
    X_train_augmented.append(image_augment(X_train_scaled[row],x=0,y=1))
    X_train_augmented.append(image_augment(X_train_scaled[row],x=0,y=-1))
    a = [y_train[row] for i in range(4)]
    y_train_augmented.append(a)

y_train_augmented_broken = []

#[]
#
for row in y_train_augmented:
    for val in row:
        y_train_augmented_broken.append(val)
#[]
#
len(X_train_augmented) == len(y_train_augmented_broken)
len(X_train) *4 == len(X_train_augmented)
len(y_train) *4 == len(y_train_augmented_broken)

#[]
#
X_train_1 = X_train_scaled.copy()
y_train_1 = y_train.copy()
X_train_augmented = np.concatenate((X_train_1,X_train_augmented))
y_train_augmented = np.concatenate((y_train_1,y_train_augmented_broken))

#[]
# checks on data
X_train_augmented[-1]

#[]
#
X_train_augmented[1]

#[]
#
plot_digit(X_train_augmented[1])
plt.show()

#[]
#
y_train_augmented[1]
#[]
#
plot_digit(X_train_augmented[-1])
plt.show()


#[]
#
y_train_augmented[-1]


#[]
# these
# ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
#  warnings.warn(
sgd_y_train_predictions_scaled_augmented = cross_val_predict(sgd_clf, X_train_augmented, y_train_augmented, cv=3, method='predict')


#[]
#
conf_mx_augmented = confusion_matrix(y_train_augmented, sgd_y_train_predictions_scaled_augmented, normalize='true')
np.fill_diagonal(conf_mx_augmented,0)
plt.matshow(conf_mx_augmented, cmap=plt.cm.gray)
plt.show()

#[]
#
sgd_scores_scaled_augmented = cross_val_score(sgd_clf, X_train_augmented, y_train_augmented, cv=3, scoring='accuracy')

#[]
# lets try a different model and see how it stacks but against our sgd model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_neighbors':[2,3,4,5],'weights':['uniform','distance']}]
kn_clf = KNeighborsClassifier()
kn_clf_grd = GridSearchCV(kn_clf, param_grid,cv=3,return_train_score=True,verbose=3)
#{]
#
kn_clf_grd.fit(X_train_augmented, y_train_augmented)
#[]
#
kn_clf_grd.cv_results_.keys()

#[]
#
kn_clf_grd.best_estimator_

#[]
#
kn_clf_grd.best_params_

#[]
#




# ***additions***
# shuffle data after having augmented it
# try np along the axis as a method for transforming multiple values in an array
# try np.array to create an array
# choose a model finally to make predictions on the test set
# cross_val_score(best_estimator, X_train_augmented, y_train_augmented, cv=3, scoring='accuracy')
