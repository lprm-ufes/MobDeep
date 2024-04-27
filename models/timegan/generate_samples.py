import timegan_utils as u
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-d', '--dataset', help="Real dataset")
parse.add_argument('-p', '--path_to_exp', help="Path to a experiment")
parse.add_argument('-n', '--n_samples', help="Number of samples to be generated", type=int)

args = parse.parse_args()

real_data =  u.np.load(args.dataset)
path_to_exp = args.path_to_exp
n_samples = args.n_samples

u.generate_timegan_samples(real_data, path_to_exp, n_samples=n_samples)