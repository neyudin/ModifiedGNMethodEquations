from oracles import eps, lim_val
import numpy as np


def psi(F, dF, x, L, tau, y):
    """
    Local model \psi_{x, L, \tau}(y) and \hat{\psi}_{x, L, \tau}(y, B) evaluated at point y.
    Parameters
    ----------
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    x : array_like
        Anchor point for the local model.
    L : float
        The estimate of local Lipschitz constant.
    tau : float
        The hyperparameter of local model.
    y : array_like
        The evaluation point for local model.
    Returns
    -------
    float
        The value of local model evaluated at point y.
    """
    return tau / 2.0 + L * np.sum(np.square(y - x)) / 2.0 +\
        np.sum(np.square(F + np.dot(dF, y - x))) / (2.0 * tau)


def factor_step_probe(F, dF, dF2=None):
    """
    Factor computation of the next point in optimization procedure using spectral decomposition and
    Sherman-Morrison-Woodbury formula.
    Parameters
    ----------
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    dF2 : array_like, default=None
        If not None, the doubly stochastic step is used and dF2 is tracted as independently
        sampled jacobian.
    Returns
    -------
    Tuple
        The tuple of factors for fast computation of the optimization step:
        Lambda, Q, ... and other factors
        Lambda : array_like
            The diagonal matrix of eigenvalues of hessian-like matrix.
        Q : array_like
            The unitary matrix of eigenvectors for corresponding eigenvalues.
    """
    m, n = dF.shape
    if m > n:
        if dF2 is None:
            Lambda, Q = np.linalg.eigh(np.dot(dF.T, dF))
        else:
            Lambda, Q = np.linalg.eigh(np.dot(dF2.T, dF2))
        return Lambda, Q, np.dot(Q.T, np.dot(dF.T, F))
    if dF2 is None:
        Lambda, Q = np.linalg.eigh(np.dot(dF, dF.T))
        return Lambda, Q, Lambda * np.dot(Q.T, F)
    Lambda, Q = np.linalg.eigh(np.dot(dF2, dF2.T))
    return Lambda, Q, np.dot(dF.T, F), np.dot(dF2.T, Q), np.dot(Q.T, np.dot(dF2, np.dot(dF.T, F)))


def probe_x(x, eta, B, v):
    """
    Computation of the next point in optimization procedure: x - eta * B^{-1}v.
    Parameters
    ----------
    x : array_like
        Current optimizable point in the procedure.
    eta : float
        The step scale.
    B : array_like
        Hessian-like matrix evaluated at x.
    v : array_like
        Gradient of 0.5 * f_2(x) evaluated at x.
    Returns
    -------
    array_like
        The next optimizable point.
    """
    return x - eta * np.dot(np.linalg.inv(np.clip(B, a_min=-lim_val, a_max=lim_val)), v)


def fast_probe_x(x, eta, tauL, F, dF, Lambda, Q, factored_QF, dF2=None):
    """
    Computation of the next point in optimization procedure using spectral decomposition and
    Sherman-Morrison-Woodbury formula.
    Parameters
    ----------
    x : array_like
        Current optimizable point in the procedure.
    eta : float
        The step scale.
    tauL : float
        The value of \tau L.
    F : array_like
        The system of equations evaluated at point x.
    dF : array_like
        The jacobian of system of equations evaluated at point x.
    Lambda : array_like
        The diagonal matrix of eigenvalues of hessian-like matrix.
    Q : array_like
        The unitary matrix of eigenvectors for corresponding eigenvalues.
    factored_QF : tuple
        The tuple of matrices and vectors from factorization of computation of the next point.
    dF2 : array_like, default=None
        If not None, the doubly stochastic step is used and dF2 is tracted as independently
        sampled jacobian.
    Returns
    -------
    array_like
        The next optimizable point.
    """
    m, n = dF.shape
    if m > n:
        return x - eta * np.dot(Q, factored_QF[0] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))
    if dF2 is None:
        return x - eta * np.dot(
            dF.T, F - np.dot(Q, factored_QF[0] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))) /\
            np.clip(tauL, a_min=eps, a_max=lim_val)
    return x - eta * (
        factored_QF[0] - np.dot(
            factored_QF[1], factored_QF[2] / np.clip(Lambda + tauL, a_min=eps, a_max=lim_val))) /\
            np.clip(tauL, a_min=eps, a_max=lim_val)

