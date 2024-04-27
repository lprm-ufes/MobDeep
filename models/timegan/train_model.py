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
import utils as u
from timegan.model import TimeGAN

def train(X, num_steps, hidden_dimensions, noise_dimensions, dimensions, learning_rates, batch_sizes, project_nm, exp_nm, runs=[], scaler=None, timestamps=[], connect_wb=False):
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
    c = 0
    for nsteps, h, nd, d, lr, bs in zip(num_steps, hidden_dimensions, noise_dimensions, dimensions, learning_rates, batch_sizes):
        # nsteps, h, nd, d, lr, bs = params
        run = runs[c]
        tf.keras.backend.clear_session()
        # rnn = u.train_rnn(X, X.shape[1])
        # u1 = u.print_memory_usage()
        gan_args_m1, seq_len_m1, n_seq_m1, hidden_dim_m1, gamma_m1, log_step_m1  = u.set_params(X, h, 1, nd, d, bs, 100, lr)
        # training
        synth = TimeGAN(model_parameters=gan_args_m1, hidden_dim=hidden_dim_m1, seq_len=seq_len_m1, n_seq=n_seq_m1, gamma=gamma_m1)
        u.creat_project_dir(project_nm, run)
        synth.train(X, train_steps=nsteps)
        synth.save(f"trainings/{project_nm}/{run}/model.pkl")
        title = f"hidden_dim:{h}|dim:{d}|lr:{lr}|noise_dim:{nd}|bs:{bs}|nsteps:{nsteps}|pn:{project_nm}|run:{run}"
        figname = u.get_figname(run)  
        run+=1
        print ("Pausing execution for 10s")
        gc.collect()
        del synth                            
        time.sleep(10)
        c+=1

def get_params(runs=[]):
    columns = ["num_steps", "hidden_dimensions", "noise_dimensions", "dimensions", "learning_rates", "batch_sizes"]
    df = pd.read_csv("trainings/experiments_tim.csv")
    params = [df.loc[runs][c].values for c in columns]
    return params

if __name__=='__main__':
    args = helper.parse_arguments()
    X, scaler = u.get_dataset(args.dataset, args.timesteps)
    timestamps = u.get_timestamps(args.timestamps)

    num_steps, hidden_dimensions, noise_dimensions, dimensions, learning_rates, batch_sizes = get_params(args.runs)

    train(X, num_steps, hidden_dimensions, noise_dimensions, dimensions, learning_rates, batch_sizes, project_nm=args.project_name, exp_nm=args.exp_name,
        runs=args.runs, scaler=scaler, timestamps=timestamps, connect_wb=args.connect_wandb)
