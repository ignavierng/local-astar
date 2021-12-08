import numpy as np
from sklearn.covariance import graphical_lasso


def glasso(X, l1_lambda=0.01, max_iter=1000):
    cov_emp = np.cov(X.T, bias=False)
    _, inv_cov_est = graphical_lasso(cov_emp, alpha=l1_lambda, max_iter=max_iter)
    return inv_cov_est
