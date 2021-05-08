from optimizers import *
import gc


def experiment_runner(args, x_0_dict):
    """
    Runner routine which performs the whole experiment set.
    Parameters
    ----------
    args : populated namespace object from ArgumentParser
        The system of equations evaluated at point x.
    x_0_dict : dict
        The dictionary of initial points x.
    Returns
    -------
    dict
        Aggregated experiment data.
    """
    gc.enable()
    gc.collect()
    
    tau_const_list = [None] + args.tau_const_list
    
    exp_res_dict = dict()
    
    exp_res_dict['DetGNM'] = dict()
    for oracle_class, name in [(NesterovSkokovOracle, 'Nesterov-Skokov'), (HatOracle, 'Hat'), (PLOracle, 'PL')]:
        if args.verbose:
            print('Oracle:', name)
        exp_res_dict['DetGNM'][name] = dict()
        for n in args.n_dims:
            if args.verbose:
                print('    n:', n)
            exp_res_dict['DetGNM'][name][n] = dict()
            for tau_const in tau_const_list:
                tau = 'GNM' if tau_const is None else tau_const
                if args.verbose:
                    print('        tau:', tau)
                exp_res_dict['DetGNM'][name][n][tau] = dict()
                for i in range(args.n_starts):
                    if args.verbose:
                        print('            start #:', i + 1)
                    _, f_vals, nabla_f_2_norm_vals, _, _ = DetGNM(oracle_class(n), args.N_iter, x_0_dict[n][i], args.L_0, True, tau_const)
                    exp_res_dict['DetGNM'][name][n][tau][i] = {'f_vals': f_vals, 'nabla_f_2_norm_vals': nabla_f_2_norm_vals}
                    del _, f_vals, nabla_f_2_norm_vals
                    gc.collect()
    
    return exp_res_dict

