import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys

def load_data(return_df=False):
    
    # generate data
    img = Image.open(str(sys.argv[1]))

    width, height = img.size
    print(width, height)


    img_pixels = []
    for y in range(height):
      for x in range(width):
        img_pixels.append(img.getpixel((x,y)))

    img_pixels = np.array(img_pixels)
    img_matrix = img_pixels.reshape([width,height])
    scatter_x = []
    scatter_y = []

    for y in range(height):
      for x in range(width):

        if img_matrix[x,y] > 1000:
            scatter_x.append(x)
            scatter_y.append(y)

    np_scatter_x = np.array(scatter_x)
    np_scatter_y = np.array(scatter_y)

    X = np.vstack([np_scatter_x, np_scatter_y]).T

    X = np_scatter_x
    y = np_scatter_y

    return X, y


def load_RM_MEDV():
    df = load_data(return_df=True)
    return df.RM.values, df.MEDV.values


class LinearRegression(object):
    """
    Ex.
        lr = LinearRegression()
        lr = lr.fit(X, y)
        y_hat = lr.predict(X)
        print(lr.score(X, y))
    """
    def __init__(self, add_const: bool=True):
        self.add_const = add_const

        self.weights = None

    def predict(self, X: np.ndarray):
        if self.weights is None:
            raise ValueError("run .fit() before prediction")

        if self.add_const:
            X = self._add_const(X)

        y_hat = X @ self.weights
        return y_hat

    def fit(self, X, y):
        self.weights = self._least_squares(X, y)
        return self

    def _least_squares(self, X: np.ndarray, y: np.ndarray):
        if self.add_const:
            X = self._add_const(X)

        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta_hat

    def _add_const(self, X):
        X_with_ones = np.c_[np.tile(1, len(X)), X]
        return X_with_ones

    def score(self, X, y) -> float:
        y_hat = self.predict(X)
        u = ((y - y_hat)**2).sum()
        v = ((y - y.mean())**2).sum()
        R_squared = 1 - u/v
        return R_squared

    def get_params(self):
        return self.weights

    def set_params(self, weights):
        self.weights = weights

class RANSACRegressor(object):
    """
    Ex.
        X, y = load_RM_MEDV()
        lr = LinearRegression()
        ransac = RANSACRegressor(base_estimator=lr, min_samples=10, residual_threshold=3, max_trials=10)
        ransac = ransac.fit(X, y)
        ransac.best_model.predict(X)
    """
    def __init__(self, base_estimator=None, min_samples=None,
                 residual_threshold=None, max_trials=100):

        self.base_estimator = base_estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold # to decide inliners
        self.max_trials = max_trials

    def fit(self, X, y):
        if self.base_estimator is None:
            raise ValueError("set .base_estimator")

        if self.min_samples is None:
            # assume linear model by default
            min_samples = X.shape[1] + 1

        if self.residual_threshold is None:
            raise ValueError("set .residual_threshold")

        if self.max_trials is None:
            raise ValueError("set .max_trials")

        model_list = []
        score_list = []
        inlier_idxs_list = []
        n_inliers_subset_list = []
        subset_index_list = []

        i_trial = 0

        sample_idxs = np.arange(X.shape[0])

        while i_trial < self.max_trials:
            print(i_trial, "step")

            estimator = copy.deepcopy(self.base_estimator)

            # random sampling
            subset_idxs = np.random.choice(len(X), self.min_samples, replace=False)
            X_subset = X[subset_idxs]
            y_subset = y[subset_idxs]

            # fit model to subset data
            estimator = estimator.fit(X_subset, y_subset)

            # predict all data and calc resdiuals
            y_hat = estimator.predict(X)
            residuals = self._absolute_loss(y, y_hat)

            # calc inlier
            inlier_mask_subset = residuals < self.residual_threshold
            inlier_mask_subset = inlier_mask_subset.flatten()
            n_inliers_subset = np.sum(inlier_mask_subset)

            inlier_idxs = sample_idxs[inlier_mask_subset]
            X_inlier_subset = X[inlier_idxs]
            y_inlier_subset = y[inlier_idxs]

            # calc score of inlier data
            score_subset = estimator.score(X_inlier_subset, y_inlier_subset)

            # save models
            model_list.append(estimator)
            score_list.append(score_subset)
            inlier_idxs_list.append(inlier_idxs)

            n_inliers_subset_list.append(n_inliers_subset)
            subset_index_list.append(subset_idxs)

            i_trial += 1

        # save best model
        best_index = np.argmax(score_list)
        self.best_index = best_index
        self.best_model = model_list[best_index]
        self.best_inlier_idxs = inlier_idxs_list[best_index]

        # save log
        self.model_list = model_list
        self.score_list = score_list
        self.inlier_idxs_list = inlier_idxs_list
        self.n_inliers_subset_list = n_inliers_subset_list
        self.subset_index_list = subset_index_list

        return self

    def _absolute_loss(self, y, y_hat):
        loss = np.abs(y - y_hat)
        return loss

X, y = load_RM_MEDV()
lr = LinearRegression()
ransac = RANSACRegressor(base_estimator=lr, min_samples=10, residual_threshold=3, max_trials=10)
ransac = ransac.fit(X, y)