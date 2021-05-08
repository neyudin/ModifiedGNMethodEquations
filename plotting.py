import numpy as np
import matplotlib.pyplot as plt
import warnings
from oracles import eps


warnings.filterwarnings('ignore')


import seaborn as sns


"""
-----------------------------------------------------------------
Mapping between filename and figure                              
-----------------------------------------------------------------
                    filename                      | fugure name  
-----------------------------------------------------------------
GNM_perf_grad_func.eps                            |   Figure 1   
GNM_perf_val_func.eps                             |   Figure 2   
-----------------------------------------------------------------
"""


def plot_experiments_results(exp_res_dict, args):
    """
    Plotting routine which draws results of the whole experiment set.
    Parameters
    ----------
    exp_res_dict : dict
        The whole infographics of the experiments.
    args : populated namespace object from ArgumentParser
        The system of equations evaluated at point x.
    Returns
    -------
    None
    """
    tau_const_list = ['GNM'] + args.tau_const_list
    
    for stat_type, stat_name in [('grad', 'nabla_f_2_norm_vals'), ('val', 'f_vals')]:
        sns.set(font_scale=1.3)
        fig, axes = plt.subplots(nrows=len(args.n_dims), ncols=3, figsize=(20, 18), sharex=True, sharey=False)
        legend_flag = False
        for col, name in enumerate(['Nesterov-Skokov', 'Hat', 'PL']):
            for row, n in enumerate(args.n_dims):
                for tau, c, marker in zip(tau_const_list, ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:gray'], ['o', '^', 'v', 's', '>', 'x', 'd', '|']):
                    data_sums = []
                    data_sizes = []
                    data_sums_of_squares = []
                    for iter_counter in range(args.N_iter):
                        for i in range(args.n_starts):
                            if iter_counter < len(exp_res_dict['DetGNM'][name][n][tau][i][stat_name]):
                                if iter_counter >= len(data_sums):
                                    data_sums.append(0.0)
                                    data_sizes.append(0)
                                    data_sums_of_squares.append(0.0)
                                data_sums[iter_counter] += exp_res_dict['DetGNM'][name][n][tau][i][stat_name][iter_counter]
                                data_sizes[iter_counter] += 1
                                data_sums_of_squares[iter_counter] +=\
                                    exp_res_dict['DetGNM'][name][n][tau][i][stat_name][iter_counter] ** 2
                    data_sizes = np.array(data_sizes)
                    data_means = np.array(data_sums) / data_sizes
                    data_stds = np.sqrt(np.abs(data_sizes * (np.array(data_sums_of_squares) / data_sizes - data_means ** 2) /\
                                        np.where(data_sizes > 1, data_sizes - 1, 1)))
                    label = r'$\tau_{k} = \hat{f}_{1}(x_{k})$' if tau == 'GNM' else r'$\tau_k = {:1.0E}$'.format(tau)
                    axes[row, col].plot(np.arange(1, data_means.size + 1), data_means, color=c, marker=marker, markevery=5,
                                        linewidth=1, ls='--', label=label)
                axes[row, col].set_yscale('log')
                if stat_type == 'grad':
                    axes[row, col].set_ylabel(r'$\|\nabla\hat{f}_{2}(x_{k})\|$', fontsize=16)
                else:
                    axes[row, col].set_ylabel(r'$\hat{f}_{1}(x_{k})$', fontsize=16)
                if row == 2:
                    axes[row, col].set_xlabel(r'Номер внешней итерации, $k$', fontsize=16)
                axes[row, col].set_title(r'${}, n = {}$'.format(name, n), fontsize=16)
                axes[row, col].axhline(y=eps, color='r', linestyle='-', linewidth=1)
                if not legend_flag:
                    legend_flag = True
                    handles, labels = axes[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.9))
        plt.savefig(fname=args.store_dir + '/GNM_perf_{}_func.eps'.format(stat_type))
        plt.close(fig)

