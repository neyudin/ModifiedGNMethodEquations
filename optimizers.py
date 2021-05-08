from opt_utils import *
from oracles import NesterovSkokovOracle, HatOracle, PLOracle, lim_val, eps


def DetGNM(oracle, N, x_0, L_0, fast_update=False, tau_const=None):
    """
    Find argminimum of f_1 using the deterministic Gauss-Newton method with exact proximal map and
    \tau_k = \hat{f}_1(x_k).
    Parameters
    ----------
    oracle : Oracle class instance
        Oracle of the optimization criterion.
    N : int
        The number of outer iterations.
    x_0 : array_like
        The initial parameter value.
    L_0 : float
        The initial value of local Lipschitz constant.
    fast_update : bool, default=True
        If true, every step is computed using the factor_step_probe and fast_probe_x functions,
        otherwise only probe_x is used.
    tau_const : float, default=None
        If not None, then the constant value is used for tau equal tau_const.
    Returns
    -------
    x : array_like
        The approximated argminimum.
    f_vals : array_like
        The list of \hat{f}_1(x_k) values at each iteration.
    nabla_f_2_norm_vals : array_like
        The list of \|\nabla\hat{f}_2(x_k)\| values at each iteration.
    nabla_f_2_vals : array_like
        The list of \nabla\hat{f}_2(x_k) values at each iteration.
    n_inner_iters : array_like
        The list of numbers of inner iterations per each outer one.
    """
    f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters = [], [], [], []
    x = x_0.copy()
    L = L_0
    for i in range(N):
        tau = (oracle.f_1(x) / np.sqrt(oracle.shape[0])) if tau_const is None else tau_const
        if tau < eps:
            break
        F = oracle.F(x) / np.sqrt(oracle.shape[0])
        dF = oracle.dF(x) / np.sqrt(oracle.shape[0])
        
        if fast_update:
            Lambda, Q, *factored_QF = factor_step_probe(F, dF)
            tmp_x = fast_probe_x(x, 1.0, tau * L, F, dF, Lambda, Q, factored_QF)
        else:
            dFTdF = np.dot(dF.T, dF)
            v = np.dot(dF.T, F)
            try:
                tmp_x = probe_x(x, 1.0, dFTdF + tau * L * np.eye(x.size), v)
            except np.linalg.LinAlgError as err:
                print('Singular matrix encountered: {}!'.format(str(err)))
                tmp_x = probe_x(x, 1.0, tau * L * np.eye(x.size), v)
        
        n = 1
        while oracle.f_1(tmp_x) / np.sqrt(oracle.shape[0]) > psi(F, dF, x, L, tau, tmp_x):
            L *= 2.0
            
            if fast_update:
                tmp_x = fast_probe_x(x, 1.0, tau * L, F, dF, Lambda, Q, factored_QF)
            else:
                try:
                    tmp_x = probe_x(x, 1.0, dFTdF + tau * L * np.eye(x.size), v)
                except np.linalg.LinAlgError as err:
                    print('Singular matrix encountered: {}!'.format(str(err)))
                    tmp_x = probe_x(x, 1.0, tau * L * np.eye(x.size), v)
            
            n += 1
        L = max(L / 2.0, L_0)
        x = tmp_x.copy()
        
        f_vals.append(oracle.f_1(x) / np.sqrt(oracle.shape[0]))
        nabla_f_2_vals.append(oracle.nabla_f_2(x) / oracle.shape[0])
        nabla_f_2_norm_vals.append(np.linalg.norm(nabla_f_2_vals[-1]))
        n_inner_iters.append(n)
        if nabla_f_2_norm_vals[-1] < eps:
            break
    
    return x, f_vals, nabla_f_2_norm_vals, nabla_f_2_vals, n_inner_iters

