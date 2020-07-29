""" File to prepare images for better analysis (only for improving performance of ML model)"""
from skimage import filters
from skimage.measure import regionprops
from scipy import ndimage

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class ImageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=50, center=True):
        self.threshold = threshold
        self.center = center

    def fit(self, X, y=None):
        """Could add a method here to fill in pixels with surrounding median
        so that the model could pick up digits that are thinner"""
        return self

    def transform(self, image, labels=None):
        """Transform images ready to be fed into model for prediction"""

        X = contrast(rgb2gray(np.array(image)))  # Image to a numpy array, grayscale + increase black white contrast
        if self.threshold != 0:                  # If threshold is not 0, make all values < threshold 0.
            X = np.array([0 if a_ < self.threshold else a_ for a_ in X.reshape([-1])]).reshape((28, 28))
        X = invert(X)                            # Invert colors (so digit pixels are in black)
        if self.threshold != 0:                  # Now make all whites white (value 0 -> removes background off whites)
            X = np.array([0 if a_ < self.threshold else a_ for a_ in X.reshape([-1])]).reshape((28, 28))
        if self.center:                          # Shift image if needed (to center digit)
            x, y = centre_mass(X)                # Weighted by value (i.e. not white) centre of mass of image
            dx, dy = 14 - int(x), 14 - int(y)
            X = shift_image(X, dx=dx, dy=dy)     # Shifts image dx, dy to image centre (14, 14) from com (x, y)
        return X.reshape([-1])                   # Return data in format models can 'read'


def rgb2gray(rgb):
    """RGB to grayscale a.k.a (m, n, 3) shape array to (m, n, 1)"""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def invert(arr):
    """Invert black and whites"""
    return int(np.amax(arr))*np.ones((28, 28)) - arr


def contrast(X):
    """Boost the contrast"""
    X_boosted = (255 / 1.5) * (X / (255 / 1.5)) ** 2
    return X_boosted


def centre_mass(im):
    """
    Finds the 'centre of mass' of the image.
    Weight assigned according to pixel values.

    returns a position
    """
    threshold_value = filters.threshold_otsu(im)
    labeled_foreground = (im > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, im)
    center_of_mass = properties[0].centroid
    weighted_center_of_mass = properties[0].weighted_centroid

    return weighted_center_of_mass


def shift_image(X, dx=0, dy=0):
    image = X.reshape(28, 28)
    im = ndimage.shift(image, [dy, dx], output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    return im.reshape([-1])


def four_image_shift(im):
    """
    Shifts one instance on the MNIST data one pixel
    in each direction to artificially grow the data
    set (data augmentation). Test, to see if it improves
    the models predictive power
    """
    moves = [-1, 1]
    ims = [im]
    for move in moves:
        ims.append(shift_image(im, dx=move))
        ims.append(shift_image(im, dy=move))

    return ims


def append_new_images(data, labels):
    x_aug = [image for image in data]
    y_aug = [label for label in labels]
    for im, lb in zip(data, labels):
        shifts = four_image_shift(im)
        for shifted_image in shifts:
            x_aug.append(shifted_image)
            y_aug.append(lb)
    return x_aug, y_aug







