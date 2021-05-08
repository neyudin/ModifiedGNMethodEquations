import numpy as np


lim_val = 1e+12 # Maximal absolute value of variables in experiments
eps = 1e-6 # Tolerance of computations


class Oracle(object):
    """
    Base class for storing nonlinear systems of equations and to work with it.
    Attributes
    ----------
    _m : int
        The number of equations.
    _n : int
        The number of parameters.
    """
    _m, _n = None, None
    
    def f_1(self, x, idxs=None):
        """
        f_1(x) --- optimitzation criterion.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of f_1
        Returns
        -------
        float
            The function f_1 value evaluated at x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        return np.clip(np.sqrt(np.sum(np.square(self.F(x, idxs)))), a_min=0, a_max=lim_val)
    
    def f_2(self, x, idxs=None):
        """
        f_2(x) = (f_1(x))^2 --- optimitzation criterion.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of f_2
        Returns
        -------
        float
            The function f_2 value evaluated at x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        return np.clip(np.sum(np.square(self.F(x, idxs))), a_min=0, a_max=lim_val)
    
    def nabla_f_2(self, x, idxs=None):
        """
        \nabla f_2(x) = \nabla (f_1(x))^2 --- gradient of the optimitzation criterion.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of \nabla f_2
        Returns
        -------
        float
            The value of \nabla f_2 evaluated at x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        return np.clip(2.0 * np.dot(self.dF(x, idxs).T, self.F(x, idxs)), a_min=-lim_val, a_max=lim_val)
    
    def F(self, x, idxs=None):
        """
        F(x) --- system of equations evaluated at point x.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of F
        Raises
        ------
        NotImplementedError
            The method should be implemented in children classes.
        """
        raise NotImplementedError
    
    def dF(self, x, idxs=None):
        """
        dF(x) --- jacobian of system of equations evaluated at point x.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of dF
        Raises
        ------
        NotImplementedError
            The method should be implemented in children classes.
        """
        raise NotImplementedError
    
    @property
    def shape(self):
        """
        Main property of systems of equations, this shape is equal to the jacobian shape.
        Returns
        -------
        tuple
            Tuple of two elements. The first one stands for the number of equations,
            the second one describes the number of parameters.
        """
        return self._m, self._n


class NesterovSkokovOracle(Oracle):
    """
    Nesterov--Skokov function f(x) represented as system of partial derivatives: F(x) = \nabla f(x).
    f(x) = 0.25 * (x_1 - 1)^2 + \sum_{i = 1}^{n - 1}(x_{i + 1} - 2x_{i}^2 + 1)^2
    x = (x_1, ..., x_n).
    Attributes
    ----------
    _m : int
        The number of equations.
    _n : int
        The number of parameters.
    """
    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            The number of optimizable float parameters, _m = n, _n = n.
        """
        self._m, self._n = n, n
    
    def F(self, x, idxs=None):
        """
        F(x) --- system of equations evaluated at point x.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of F
        Returns
        -------
        array_like
            Float values of the system of equations at point x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        F_vals = np.zeros(self._m)
        F_vals[0] += (x[0] - 1.0) / 2.0
        F_vals[1:] += -4.0 * (x[:-1] ** 2) + 2.0 * x[1:] + 2.0
        F_vals[:-1] += 16.0 * (x[:-1] ** 3) - 8.0 * x[:-1] * x[1:] - 8.0 * x[:-1]
        F_vals = np.clip(F_vals, a_min=-lim_val, a_max=lim_val)
        if idxs is None:
            return F_vals
        return F_vals[idxs]
    
    def dF(self, x, idxs=None):
        """
        dF(x) --- jacobian of system of equations evaluated at point x.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of dF
        Returns
        -------
        array_like
            Jacobian the system of equations at point x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        dF_vals = np.zeros((self._m, self._n))
        coords = np.arange(self._m)
        
        main_diag_vals = np.zeros(self._m)
        main_diag_vals[0] += 0.5
        main_diag_vals[1:] += 2.0
        main_diag_vals[:-1] += 48.0 * (x[:-1] ** 2) - 8.0 * x[1:] - 8.0
        
        dF_vals[coords, coords] += main_diag_vals
        dF_vals[coords[1:], coords[1:] - 1] += -8.0 * x[:-1]
        dF_vals[coords[:-1], coords[:-1] + 1] += -8.0 * x[:-1]
        
        dF_vals = np.clip(dF_vals, a_min=-lim_val, a_max=lim_val)
        if idxs is None:
            return dF_vals
        return dF_vals[idxs]


class HatOracle(Oracle):
    """
    f(x) = (\|x\|^2 - 1)^2 represented as system of partial derivatives: F(x) = \nabla f(x).
    Attributes
    ----------
    _m : int
        The number of equations.
    _n : int
        The number of parameters.
    """
    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            The number of optimizable float parameters, _m = n, _n = n.
        """
        self._m, self._n = n, n
    
    def F(self, x, idxs=None):
        """
        F(x) --- system of equations evaluated at point x.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of F
        Returns
        -------
        array_like
            Float values of the system of equations at point x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        if idxs is None:
            return np.clip(4 * (np.sum(x ** 2) - 1.0) * x, a_min=-lim_val, a_max=lim_val)
        return np.clip(4 * (np.sum(x ** 2) - 1.0) * x[idxs], a_min=-lim_val, a_max=lim_val)
    
    def dF(self, x, idxs=None):
        """
        dF(x) --- jacobian of system of equations evaluated at point x.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of dF
        Returns
        -------
        array_like
            Jacobian the system of equations at point x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        dF_vals = 8.0 * x * x[:, np.newaxis]
        coords = np.arange(self._m)
        dF_vals[coords, coords] += 4.0 * (np.sum(x ** 2) - 1.0)
        dF_vals = np.clip(dF_vals, a_min=-lim_val, a_max=lim_val)
        if idxs is None:
            return dF_vals
        return dF_vals[idxs]


class PLOracle(Oracle):
    """
    f(x) = \|x\|^2 + 3\sum_{i = 1}^{n}\sin^2(x_{i}) represented as
    system of partial derivatives: F(x) = \nabla f(x).
    Attributes
    ----------
    _m : int
        The number of equations.
    _n : int
        The number of parameters.
    """
    def __init__(self, n):
        """
        Parameters
        ----------
        n : int
            The number of optimizable float parameters, _m = n, _n = n.
        """
        self._m, self._n = n, n
    
    def F(self, x, idxs=None):
        """
        F(x) --- system of equations evaluated at point x.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of F
        Returns
        -------
        array_like
            Float values of the system of equations at point x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        if idxs is None:
            return np.clip(2.0 * x + 3.0 * np.sin(2.0 * x), a_min=-lim_val, a_max=lim_val)
        return np.clip((2.0 * x + 3.0 * np.sin(2.0 * x))[idxs], a_min=-lim_val, a_max=lim_val)
    
    def dF(self, x, idxs=None):
        """
        dF(x) --- jacobian of system of equations evaluated at point x.
        Parameters
        ----------
        x : array_like
            Array of optimizable float parameters.
        idxs : array_like, default=None
            Indexes of used functions in the random sample of dF
        Returns
        -------
        array_like
            Jacobian the system of equations at point x.
        """
        x = np.clip(x, a_min=-lim_val, a_max=lim_val)
        if idxs is None:
            return np.clip(np.diag(2.0 + 6.0 * np.cos(2.0 * x)), a_min=-lim_val, a_max=lim_val)
        return np.clip(np.diag(2.0 + 6.0 * np.cos(2.0 * x))[idxs], a_min=-lim_val, a_max=lim_val)

