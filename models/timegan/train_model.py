import gc
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #don't show tensorflow debbugs

import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import helper
import metrics as m
import timegan_utils as u
from timegan.model import TimeGAN

def train(X, num_steps, hidden_dimensions, noise_dimensions, dimensions, learning_rates, batch_sizes, exp_id=None, test_id=None):
    """
    Train TimeGAN model using parameters passed by command line

    Params:
        - num_steps:
        - hidden_dimensions
        - noise_dimensions
        - dimensions
        - learning_rates
        - batch_sizes
        - project_nm
        - exp_nm
        - run:
        - scaler:
    """
    shp = X.shape
    col0 = X.reshape(shp[0]*shp[1], shp[-1])[:, 0]
    nsteps, h, nd, d, lr, bs = [num_steps, hidden_dimensions, noise_dimensions, dimensions, learning_rates, batch_sizes]
    # nsteps, h, nd, d, lr, bs = params
    tf.keras.backend.clear_session()
    # rnn = u.train_rnn(X, X.shape[1])
    # u1 = u.print_memory_usage()
    gan_args_m1, seq_len_m1, n_seq_m1, hidden_dim_m1, gamma_m1, log_step_m1  = u.set_params(X, h, 1, nd, d, bs, 100, lr)
    # training
    synth = TimeGAN(model_parameters=gan_args_m1, hidden_dim=hidden_dim_m1, seq_len=seq_len_m1, n_seq=n_seq_m1, gamma=gamma_m1)
    synth.train(X, train_steps=nsteps)
    synth.save(f"experiments/{exp_id}/{test_id}/models/model.pkl")  
    print ("Pausing execution for 10s")
    gc.collect()
    del synth                            
    time.sleep(10)

def get_params(runs=[]):
    columns = ["num_steps", "hidden_dimensions", "noise_dimensions", "dimensions", "learning_rates", "batch_sizes"]
    df = pd.read_csv("trainings/experiments_tim.csv")
    params = [df.loc[runs][c].values for c in columns]
    return params

if __name__=='__main__':
    args = helper.parse_arguments()
    X, scaler = u.get_dataset(args.dataset, args.timesteps, args.scale)
    timestamps = u.get_timestamps(args.timestamps)
    num_steps = args.num_steps
    hidden_dimensions = args.hidden_dimension    
    noise_dimensions = args.noise_dimension
    dimensions = args.dimension
    learning_rates = args.learning_rate
    batch_sizes = args.batch_size

    train(X, num_steps, hidden_dimensions, noise_dimensions, dimensions, learning_rates, batch_sizes, exp_id=args.exp_id, test_id=args.test_id)
