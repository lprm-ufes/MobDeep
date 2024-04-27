import argparse

def parse_arguments():
    parse = argparse.ArgumentParser(description='Train TimeGan models')
    parse.add_argument('-d', '--dataset', help='training dataset')
    parse.add_argument('-ts','--timestamps', help='dateset of dates')
    parse.add_argument('-t', '--timesteps', help='dataset timesteps (n, t, v)', type=int)
    parse.add_argument('-m', '--rnn_model', help='train rnn model on the fly', action='store_true', default=False)    

    # model hyperparameters
    # parse.add_argument('-n', '--num_steps', help='list of num_steps', nargs='+', type=int)
    # parse.add_argument('-hd','--hidden_dimensions', help='list of hidden_dimensios', nargs='+', type=int)
    # parse.add_argument('-nd','--noise_dimensions', help='list of noise_dimensios', nargs='+', type=int)
    # parse.add_argument('-dm','--dimensions',help='list of dims', nargs='+', type=int)
    # parse.add_argument('-l', '--learning_rates', help='list of learning_rates', nargs='+', type=float)
    # parse.add_argument('-bs','--batch_sizes', help='list of batch_sizes', nargs='+', type=int)
    
    # Weights&Biases
    parse.add_argument('-w', '--connect_wandb', action='store_true', help='connect to weights&biases dashboard')
    parse.add_argument('-p', '--project_name', type=str, help='set project name')
    parse.add_argument('-e', '--exp_name', type=str, help='set the experiment name')
    parse.add_argument('-r', '--runs', type=int, nargs='+', required=True, help='Experiments number')

    args = parse.parse_args()
    return args