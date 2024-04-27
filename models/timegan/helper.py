import argparse

def parse_arguments():
    parse = argparse.ArgumentParser(description='Train TimeGan models')
    parse.add_argument('-d', '--dataset', help='training dataset')
    parse.add_argument('-ts','--timestamps', help='dateset of dates')
    parse.add_argument('-t', '--timesteps', help='dataset timesteps (n, t, v)', type=int)
    parse.add_argument('-m', '--rnn_model', help='train rnn model on the fly', action='store_true', default=False)    

    # model hyperparameters
    parse.add_argument('-n', '--num_steps', help='num_steps', type=int)
    parse.add_argument('-hd','--hidden_dimension', help='hidden_dimension', type=int)
    parse.add_argument('-nd','--noise_dimension', help='noise_dimensio', type=int)
    parse.add_argument('-dm','--dimension', help='dims', type=int)
    parse.add_argument('-l', '--learning_rate', help='learning_rate', type=float)
    parse.add_argument('-bs','--batch_size', help='batch_size', type=int)

    #
    parse.add_argument('-ti', '--test_id', help='Test id')
    parse.add_argument('-ei', '--exp_id', help='Exp id')
    parse.add_argument('-s', '--scale', action='store_true', help='Whether to scale the data before training (not necessarry if the data is already scaled.)')

    args = parse.parse_args()
    return args