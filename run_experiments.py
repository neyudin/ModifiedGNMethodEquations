from benchmark_utils import experiment_runner
from plotting import plot_experiments_results
import numpy as np

from pathlib import Path
import argparse
import time
import pickle as pkl


class Store_as_array(argparse._StoreAction):
    """
    Helper class to convert data stream into numpy array.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


class Store_as_list(argparse._StoreAction):
    """
    Helper class to convert data stream into list.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        values = list(values)
        return super().__call__(parser, namespace, values, option_string)


parser = argparse.ArgumentParser('Flex GNM')
parser.add_argument('--N_iter', type=int, default=100, help='The number of iterations to run Gauss-Newton method.')
parser.add_argument('--seed', type=int, default=617, help='The random seed.')
parser.add_argument(
    '--n_starts', type=int, default=5, help='The number of random samples for each combination of hyperparameters.')
parser.add_argument(
    '--verbose', type=bool, default=False, help='Whether to print auxiliary messages throughout the whole experiment.')
parser.add_argument(
    '--store_dir', type=str, default='.', help="The directory to store experiments' results.")
parser.add_argument(
    '--n_dims', action=Store_as_array, type=int, nargs='+', default=np.array([10, 100, 1000]),
    help='The list of numbers of parameters.')
parser.add_argument('--stoch_n_dim', type=int, default=1000, help='The number of parameters in stochastic settings.')
parser.add_argument(
    '--b_sizes', action=Store_as_array, type=int, nargs='+', default=np.array([1, 10, 100, 1000]),
    help='The list of the batch sizes, each batch size is no greater than the last value in n_dims.')
parser.add_argument(
    '--eta_0_list', action=Store_as_array, type=float, nargs='+',
    default=np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
    help='The list of step scales in the doubly stochastic regime.')
parser.add_argument(
    '--eta_list', action=Store_as_array, type=float, nargs='+',
    default=np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
    help='The list of step scales.')
parser.add_argument('--l_0', type=float, default=1.0,
                    help='Initial estimate of the local Lipschitz constant for \varphi local model.')
parser.add_argument('--L_0', type=float, default=1.0,
                    help='Initial estimate of the local Lipschitz constant for \psi local model.')
parser.add_argument(
    '--tauL_list', action=Store_as_list, type=str, nargs='+', default=[1e-4, 1.0, 1e3, 1e6, '+inf'],
    help='The values of \tau L products.')
parser.add_argument(
    '--tau_const_list', action=Store_as_list, type=float, nargs='+', default=[1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6],
    help='The values of \tau const.')


args = parser.parse_args()


if __name__ == '__main__':
    start = time.time() # Start time measurement for experiments.
    Path(args.store_dir).mkdir(parents=True, exist_ok=True) # Create directory store_dir if it does not exist.
    
    np.random.seed(args.seed) # The random seed specification for reproducibility.
    x_0_dict = {n: np.random.randn(args.n_starts, n) for n in args.n_dims} # The dictionary of the initial values of parameters.
    
    exp_res_dict = experiment_runner(args, x_0_dict) # Run experiments.
    pkl.dump(exp_res_dict, open(args.store_dir + '/flex_gnm_experiments_results.pkl', 'wb')) # Save infographics.
    
    plot_experiments_results(exp_res_dict, args) # Plot results.
    
    start = time.time() - start # End time measurement for experiments.
    print(
        'Elapsed runtime: {} day(s), {} hour(s), {} minute(s), {} second(s)'.format(
            int(start // 86400), int(start // 3600 % 24), int(start // 60 % 60), int(start % 60)))

