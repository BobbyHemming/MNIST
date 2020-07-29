import os
import sys
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, '/Users/roberthemming/SourceCode/Python/Packages/timeTest')
import timer as tt
import pandas as pd
import numpy as np

from visualisation import plot_digit_prediction, plot_multiple_digits
<<<<<<< HEAD
from imprep import centre_mass, shift_image, four_image_shift, append_new_images
from modelprep import save_model, load_model

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib
import joblib
=======
from imprep import centre_mass, shift_image

from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit
from skimage.color import label2rgb
>>>>>>> 94e9c97f906374e4057eb92d2cfe2a0e1413eec7

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

<<<<<<< HEAD
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 20)
# pd.set_option('display.width', 1000)
=======
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
>>>>>>> 94e9c97f906374e4057eb92d2cfe2a0e1413eec7

PROJECT_ROOT_DIR = ""
CHAPTER_ID = "classification"


def save_im(fig_name, tight=True):
    """ Save the images to a new folder 'images' """
    file_path = os.path.join(PROJECT_ROOT_DIR, "images", fig_name + ".png")
    print("Saving figure", fig_name)
    if tight:
        plt.tight_layout()
    plt.savefig(file_path, format='png', dpi=300)


def sort_by_target(mnist, ratio):
    """ Sort the MNIST data according to the label value (takes values from 0-9) """
    split_index = int(ratio*len(mnist.target))
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:split_index])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[split_index:])]))[:, 1]
    mnist.data[:split_index] = mnist.data[reorder_train]
    mnist.target[:split_index] = mnist.target[reorder_train]
    mnist.data[split_index:] = mnist.data[reorder_test + split_index]
    mnist.target[split_index:] = mnist.target[reorder_test + split_index]


mnist_dataset = fetch_openml('mnist_784', version=1, cache=True)
mnist_dataset.target = mnist_dataset.target.astype(np.int8)  # fetch_openml() returns targets as strings
sort_by_target(mnist_dataset, (6.0/7.0))  # fetch_openml() returns an unsorted dataset
<<<<<<< HEAD
=======
print(mnist_dataset['target'], type(mnist_dataset['target']))
>>>>>>> 94e9c97f906374e4057eb92d2cfe2a0e1413eec7

mnist_df = pd.DataFrame({'data': list(mnist_dataset['data']), 'target': np.array(mnist_dataset['target'])},
                        columns=['data', 'target'])

"""Data split"""
# print(mnist_df.head())
split = StratifiedShuffleSplit(n_splits=1, test_size=(1/7.0), random_state=42)
for train_index, test_index in split.split(mnist_df['data'], mnist_df['target']):
<<<<<<< HEAD
    x, y = mnist_dataset['data'][train_index], mnist_dataset['target'][train_index]
    x_test, y_test = mnist_dataset['data'][test_index], mnist_dataset['target'][test_index]


strat_props_train = [round(i, 3) for i in (np.unique(y, return_counts=True)[1]/len(y)).tolist()]
strat_props_test = [round(m, 3) for m in (np.unique(y_test, return_counts=True)[1]/len(y_test)).tolist()]

strat_check = zip(strat_props_train, strat_props_test)
print(tuple(strat_check))
=======
    mnist_train = mnist_df.iloc[train_index]
    mnist_test = mnist_df.iloc[test_index]

# print(mnist_train.head())
# print(mnist_train.info())

print(f'Training Target data type: {type(mnist_train["target"])} '
      f'and an example: {mnist_train["target"].iloc[0]}')

x, y = mnist_train['data'].to_numpy(), mnist_train['target'].to_numpy()
x_test, y_test = mnist_test['data'].to_numpy(), mnist_test['target'].to_numpy()
>>>>>>> 94e9c97f906374e4057eb92d2cfe2a0e1413eec7


if not os.path.isfile('images/multiple_digit_example.png'):
    plt.figure(figsize=(9, 9))
    example_images = np.r_[x[:12000:600], x[13000:30600:600], x[30600:60000:590]]
    plot_multiple_digits(example_images, images_per_row=10)
    save_im("multiple_digit_example")
    plt.show()

<<<<<<< HEAD
# for a new list of images including 4 more shifted in each direction: (1 instance to 5, 3 instances to 15)
# [new_images2.append(i) for i in four_image_shift(image) for image in images]


""" Data Cleaning/Feature Engineering steps - Effectively data preprocessing before fitting the model
    This should try to maximise the performance of the ML algorithm on the training set                """

# Add custom estimators and transformers here. The custom image centering transformer was an idea for the MNIST data
# but once I built the function to find the centers I realised the digits were already centered. Not an issue as it
# wasn't a long process, but would be better if in the future I remembered to read information about dataset more
# carefully. It may save me time in the future.

pipeline = Pipeline([('std_scalar', StandardScaler())])  # Other steps can be added in here after experimentation
x_prepared = pipeline.fit_transform(x.astype(np.float64))  # The model actually does worse with scaled x values.
# scaler = StandardScaler()
# x_prepared = scaler.fit_transform(x.astype(np.float64))

""" Fit to a model """
print('Fitting a model stage')

param_grid = [{'n_neighbors': [4, 5], 'weights': ['uniform', 'distance']}]


kn_clf = KNeighborsClassifier()
if not os.path.isfile("models/grid_search.pkl"):
    print("Doing grid search...")
    grid_search = GridSearchCV(kn_clf, param_grid, cv=2, verbose=3, n_jobs=-1)
    grid_search.fit(x, y)
    save_model(grid_search, "models/grid_search")


grid_search = load_model("models/grid_search")
print('Grid Search loaded')


""" Evaluate performance and best params for K Neighbors test"""
# kn_score = cross_val_score(kn_clf, x_prepared, y, cv=3, scoring="accuracy")


# kn_clf.fit(x, y)
# # x_test_prepared = pipeline.transform(x_test)
# y_knn_pred = kn_clf.predict(x_test)
# knn_score = accuracy_score(y_test, y_knn_pred)
# print(knn_score)


print('\n GRID SEARCH RESULTS')
grid_search = load_model("models/grid_search")
print(grid_search.best_params_)
print(grid_search.best_estimator_)
print('Grid Search best score:', grid_search.best_score_)


if not os.path.isfile('models/KNeighborsClassifier.pkl'):
    kn_clf = KNeighborsClassifier(**grid_search.best_params_)
    kn_clf.fit(x, y)
    save_model(kn_clf, "models/KNeighborsClassifier")


kn_clf = load_model("models/KNeighborsClassifier")
print('\nKNeighbors Classifier loaded')

y_knn_pred = kn_clf.predict(x_test)
knn_score = accuracy_score(y_test, y_knn_pred)
print('KNN best classifier score on test set:', knn_score, '\n')


some_digit_index = 2000
print(f'\n Some digit label: {y[some_digit_index]}')
print('Model guess: ', kn_clf.predict([x[some_digit_index]]))
print(x[some_digit_index])
plt.figure(figsize=(9, 9))
plot_digit_prediction(x[some_digit_index], y[some_digit_index], kn_clf.predict([x[some_digit_index]]))
save_im("digit")
plt.show()


def aug_data():
    print('Augemnted data')
    aug = append_new_images(x, y)
    x_aug, y_aug = np.array(aug[0]), np.array(aug[1])

    if not os.path.isfile('models/KNAugmentedClassifier.pkl'):
        kn_clf_aug = KNeighborsClassifier(**grid_search.best_params_)
        kn_clf_aug.fit(x_aug, y_aug)
        save_model(kn_clf_aug, 'models/KNAugmentedClassifier')

    kn_clf_aug = load_model('models/KNAugmentedClassifier')
    y_knn_aug_pred = kn_clf.predict(x_test)
    knn_score = accuracy_score(y_test, y_knn_aug_pred)
    print('KNN best classifier (with augmented data) score on test set:', knn_score)
=======

image = x[1500].reshape(28, 28)
moves = [-1, 1]

images = [image, x[2000].reshape(28, 28), x[3000].reshape(28, 28)]


def four_image_shift(im):

    ims = [im]
    for move in moves:
        ims.append(shift_image(im, dx=move))
        ims.append(shift_image(im, dy=move))

    for image in ims:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="Greys")
        center_of_mass = centre_mass(image)
        ax.scatter(center_of_mass[1], center_of_mass[0], s=160, c='C0', marker='+')
        plt.show()

    return ims


new_images = []
new_images2 = []
for image in images:
    for i in four_image_shift(image):
        new_images.append(i)

[new_images2.append(i) for i in four_image_shift(image) for image in images]
print(len(new_images), len(new_images2))


>>>>>>> 94e9c97f906374e4057eb92d2cfe2a0e1413eec7

