""" Functions for running GLM on 2D and 3D data

Adapted from Matthew Brett's GLM intorduction and code(https://matthew-brett.github.io/teaching/glm_intro.html)

"""

import numpy as np
import numpy.linalg as npl
# LAB(begin solution)
import scipy.stats
# LAB(end solution)


def glm(Y, X):
    """ Run GLM on on data `Y` and design `X`

    Parameters
    ----------
    Y : array shape (N, V)
        1D or 2D array to fit to model with design `X`.  `Y` is column
        concatenation of V data vectors.
    X : array ahape (N, P)
        2D design matrix to fit to data `Y`.

    Returns
    -------
    B : array shape (P, V)
        parameter matrix, one column for each column in `Y`.
    sigma_2 : array shape (V,)
        unbiased estimate of variance for each column of `Y`.
    df : int
        degrees of freedom due to error.
    """
    # LAB(begin solution)
    B = npl.pinv(X).dot(Y)
    E = Y - X.dot(B)
    df = X.shape[0] - npl.matrix_rank(X)
    sigma_2 = np.sum(E ** 2, axis=0) / df
    return B, sigma_2, df

def glm_4d(Y, X):
    """ Run GLM on on 4D data `Y` and design `X`

    Parameters
    ----------
    Y : array shape (I, J, K, T)
        4D array to fit to model with design `X`.  Column vectors are vectors
        over the final length T dimension.
    X : array ahape (T, P)
        2D design matrix to fit to data `Y`.

    Returns
    -------
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.
    """
    # LAB(begin solution)
    I, J, K, T = Y.shape
    P = X.shape[1]
    Y_2d = Y.reshape((-1, T)).T
    B_2d, s_2_2d, df = glm(Y_2d, X)
    return B_2d.T.reshape((I, J, K, P)), s_2_2d.reshape((I, J, K)), df

def t_test(c, X, B, sigma_2, df):
    """ Two-tailed t-test given contrast `c`, design `X`

    Parameters
    ----------
    c : array shape (P,)
        contrast specifying conbination of parameters to test.
    X : array shape (N, P)
        design matrix.
    B : array shape (P, V)
        parameter estimates for V vectors of data.
    sigma_2 : float
        estimate for residual variance.
    df : int
        degrees of freedom due to error.

    Returns
    -------
    t : array shape (V,)
        t statistics for each data vector.
    p : array shape (V,)
        two-tailed probability value for each t statistic.
    """
    # LAB(begin solution)
    t_dist = scipy.stats.t(df=df)
    b_cov = npl.pinv(X.T.dot(X))
    t = c.dot(B) / np.sqrt(sigma_2 * c.dot(b_cov).dot(c))
    p_values = (1 - t_dist.cdf(np.abs(t))) * 2
    return t, p_values

def t_test_3d(c, X, B, sigma_2, df):
    """ Two-tailed t-test on 3D estimates given contrast `c`, design `X`

    Parameters
    ----------
    c : array shape (P,)
        contrast specifying conbination of parameters to test.
    X : array shape (N, P)
        design matrix.
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.

    Returns
    -------
    t : array shape (I, J, K)
        t statistics for each data vector.
    p : array shape (V,)
        two-tailed probability value for each t statistic.
    """
    # LAB(begin solution)
    I, J, K, P = B.shape
    t_1d, p_1d = t_test(c, X, B.reshape((-1, P)).T, sigma_2.ravel(), df)
    return t_1d.reshape((I, J, K)), p_1d.reshape((I, J, K))
