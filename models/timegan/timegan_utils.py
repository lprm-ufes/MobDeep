import datetime
import glob
import os
import time

import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from synthesizers.gan import ModelParameters
from timegan.model import TimeGAN


def print_memory_usage():
  MB = 1024*1024
  swap = psutil.swap_memory()
  ram = (psutil.virtual_memory()[3] + swap.total) / (MB)
  print ("{:.2f} Mb".format(ram))
  return ram

def pdf(d):
  hist, edges = np.histogram(d)
  pdf = hist / hist.sum()
  return pdf, edges

def cdf(pdf):  
  return np.cumsum(pdf)

#### Compare
def compare(X, title, fake, figname=None, project_nm='', run=0):
  soma = X.sum(axis=1)
  soma_fk = fake.sum(axis=1)

  fig, ax = plt.subplots(ncols=3, figsize=(15,3.5), )
  for i in range(3):
    ax[i].plot(soma[:, i], label='Real')
    ax[i].plot(soma_fk[:, i][:X.shape[0]], label='Fake')
    ax[i].legend()
    ax[i].set_xlabel("Janela de 5 minutos")
  ax[0].set_ylabel("Número de usuários")
  plt.suptitle(title)
  if figname:
    plt.savefig(f'trainings/{project_nm}/{run}/soma_{figname}.png')
  plt.close()

def plot_dist(X, title, fake, figname=None, project_nm='', run=0):
  fig, ax = plt.subplots(ncols=3, figsize=(15, 3.5))
  for i in range(3):
    r = X[:, :, i].flatten()
    f = fake[:X.shape[0], :, i].flatten()
    pdf_r, edges_r = pdf(r)
    pdf_f, edges_f = pdf(f)
    ax[i].plot(edges_r[1:], cdf(pdf_r), label='Real')
    ax[i].plot(edges_f[1:], cdf(pdf_f), label='Fake')
  ax[0].set_ylabel("CDF")
  plt.suptitle(title)
  plt.tight_layout()
  if figname:
    plt.savefig(f'trainings/{project_nm}/{run}/dist_{figname}.png')
  plt.close()

def get_sumhour(timestamps, cnts):
  dictdf = {
      'time': timestamps
  }
  for i in range(cnts.shape[-1]):
      dictdf[f'cnt_{i}'] = cnts[:, i]
  df = pd.DataFrame(dictdf)
  df['hour'] = df['time'].apply(lambda a: a.hour)
  hours = sorted(df['hour'].unique())
  cols = df.columns[1:]
  list_sums = [df[df['hour']==h][cols].sum().values for h in hours]
  list_sums = np.array(list_sums)
  return hours, list_sums

def plot_sumhour(timestamps, cnts, fake_cnts, title, figname, project_nm, run):
  cnts_copy = cnts.reshape(cnts.shape[0]*cnts.shape[1], cnts.shape[-1])
  fake_cnts = fake_cnts.reshape(fake_cnts.shape[0]*fake_cnts.shape[1], fake_cnts.shape[-1])
  hours, list_sums = get_sumhour(timestamps, cnts_copy)
  _, list_sums_fake = get_sumhour(timestamps, fake_cnts)

  fig, ax = plt.subplots(ncols=cnts_copy.shape[-1], figsize=(15, 4))
  for i in range(cnts_copy.shape[-1]):
    ax[i].plot(hours, list_sums[:, i], label='real')
    ax[i].plot(hours, list_sums_fake[:, i], label='fake')
    ax[i].set_xlabel("Hour")
  ax[0].set_ylabel("Nº users")
  plt.suptitle(title)
  plt.legend()
  plt.tight_layout()
  if figname:
    plt.savefig(f'trainings/{project_nm}/{run}/sumhour_{figname}.png')
  plt.close()

def get_figname(run):
  day = datetime.datetime.now().day
  month = datetime.datetime.now().month
  year = datetime.datetime.now().year
  hour = datetime.datetime.now().hour
  minutes = datetime.datetime.now().minute
  figname=f"{day}_{month}_{year}_{hour}_{minutes}"
  return figname  



def set_params(X, hdm, gamma, noise_dim, dim, bs, logstep, lr):
  ### definição dos hiperparâmetros
  # parâmetros constantes
  seq_len_m1 = X.shape[1]
  n_seq_m1 = X.shape[-1]

  # parâmetros para modificar
  hidden_dim_m1=hdm
  gamma_m1=gamma

  noise_dim_m1 = noise_dim
  dim_m1 = dim
  batch_size_m1 = bs

  log_step_m1 = logstep
  learning_rate_m1 = lr #3e-4

  gan_args_m1 = ModelParameters(batch_size=batch_size_m1,
                            lr=learning_rate_m1,
                            noise_dim=noise_dim_m1,
                            layers_dim=dim_m1)
  return gan_args_m1, seq_len_m1, n_seq_m1, hidden_dim_m1, gamma_m1, log_step_m1

def creat_project_dir(project_nm, run):
  path = f"trainings/{project_nm}/{run}/"
  if not os.path.isdir(path):
    os.makedirs(path)

def make_rnn(units, n_layers=3, n_steps=24, n_features=1, output_units=1, net_type='GRU'):
  model = Sequential()
  if net_type == 'GRU':
    for i in range(n_layers):
      if i == n_layers-1:
        model.add(GRU(units,input_shape=(n_steps,n_features), name="GRU_{}".format(i+1)))
      else:
        model.add(GRU(units,input_shape=(n_steps,n_features), return_sequences=True, name="GRU_{}".format(i+1)))
  else:
    for i in range(n_layers):
      if i == n_layers-1:
        model.add(LSTM(units, input_shape=(n_steps,n_features),name="LSTM_{}".format(i+1)))
      else:
        model.add(LSTM(units, input_shape=(n_steps,n_features), return_sequences=True, name="LSTM_{}".format(i+1)))
  model.add(Dense(units=output_units,name='OUT'))
  return model

def scale_data(data,nsteps, nfeatures):
  scaler = MinMaxScaler().fit(data.reshape(-1,1))
  data_scaled = scaler.transform(data.reshape(-1,1))
  data_scaled = data_scaled.reshape(len(data_scaled)//(nsteps*nfeatures),nsteps,nfeatures)
  return data_scaled

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def train_rnn(data, n_steps, scale=False):
  rnn = make_rnn(24,n_layers=2,n_steps=n_steps, n_features=data.shape[-1], output_units=data.shape[-1],net_type='LSTM')
  opt = Adam(learning_rate=5e-4)
  rnn.compile(optimizer=opt,loss='mse')
  if scale:
    train_s = scale_data(data, n_steps, data.shape[-1])
    shp_train = train_s.shape
    train_s = train_s.reshape(shp_train[0]*shp_train[1], data.shape[-1])
  else:
    train_s = data.reshape((data.shape[0]*data.shape[1], data.shape[-1]))
  X, y = split_sequence(train_s, n_steps)
  # training
  es = EarlyStopping(monitor='loss', patience=5)
  hist = rnn.fit(X, y, batch_size=64, epochs=50, callbacks=[es], verbose=0)
  return rnn

def get_residuals(real, fake):
  err = np.abs(fake - real)
  return err

def load_residual_model(folder):
  model = load_model(f'models/{folder}/rnn.h5')      
  return model

def get_dataset(file, timesteps=10, scale=False):
  data = np.load(file, )
  scaler = None
  if scale:
    scaler = MinMaxScaler().fit(data.reshape(-1,1))
    X = scaler.transform(data.reshape(-1,1))
  X = data
  if data.ndim<3:
    X = X.reshape(data.shape[0]//timesteps, timesteps, data.shape[-1])
  return X, scaler

def get_timestamps(file):
  timestamps = np.load(file, allow_pickle=True)
  return timestamps

def TSTR(real, synthetics = [], nsteps=10, normalise=False, fr=(-1,1)):
  """
  Train on synthetic test on real

  Params:
  - real: real data
  - synthetics: list of synthetic data
  - nsteps: timesteps of the real data (same as used for gan training)
  - normalise: if error should use normalised data
  """
  scaler = MinMaxScaler(feature_range=fr).fit(real.reshape(-1,1))
  if normalise:
    real = scaler.transform(real.reshape(-1,1)).reshape(real.shape)    
    X_real, y_real = split_sequence(real, nsteps)
    # y_real = scaler.transform(y_real.reshape(-1,1)).reshape(y_real.shape)
  maes = []
  for synth in synthetics:
    rnn = train_rnn(synth, nsteps) # synth is already scaled [0, 1]
    y_pred = rnn.predict(X_real, verbose=0)
    if not normalise:
      y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
    y_pred = y_pred.reshape(y_real.shape)
    mae = mean_absolute_error(y_real, y_pred)
    maes.append(mae)
  return maes

def TRTS(real, synth, rnn, y_real, nsteps=10, normalise=False, fr=(-1,1)):
  scaler = MinMaxScaler(feature_range=fr).fit(real.reshape(-1,1))
  # if normalise:
  #   real = scaler.transform(real.reshape(-1,1)).reshape(real.shape)    
  #   X_real, y_real = split_sequence(real, nsteps)
  #   # y_real = scaler.transform(y_real.reshape(-1,1)).reshape(y_real.shape)
  X_synth, y_synth = split_sequence(synth, nsteps)
  y_pred = rnn.predict(X_synth, verbose=0)
  if not normalise:
    y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
  y_pred = y_pred.reshape(y_real.shape)
  mae = mean_absolute_error(y_synth, y_pred)
  return mae

def generate_timegan_samples(real_data, path_to_exp, n_samples=50):
  print ("Opa funciona")
  max_size = real_data.shape[0]
  path_to_model = path_to_exp+"/models/model.pkl"
  path_to_samples = path_to_exp+"/synthdata"
  synth = TimeGAN.load(path_to_model)
  for i in range(n_samples):
    synth_s = synth.sample(max_size)
    synth_s = synth_s[:max_size]
    np.save(path_to_samples+f"/{i}.npy", synth_s)

def generate_timegan_best(model_path, sample_folder, real_data, n_samples=1000):
  max_size = real_data.shape[0]
  synth = TimeGAN.load(model_path)
  try:
    os.makedirs(sample_folder)
  except:
    pass
  for i in tqdm(range(n_samples)):
    if not os.path.isfile(sample_folder+f"/{i}.npy"):
      synth_s = synth.sample(max_size)
      synth_s = synth_s[:max_size]    
      np.save(sample_folder+f"/{i}.npy", synth_s)

def compute_tstr(path_samples, inicio, step, train_data,normalise, fr):
  list_mae = []
  inicio = inicio
  fim = inicio + step
  list_samples = glob.glob(path_samples)
  for i in tqdm(range(inicio, fim)):
    s = list_samples[i]
    sample = np.load(s)
    mae = TSTR(train_data, synthetics=[sample], normalise=normalise, fr=fr)[0]
    list_mae.append(mae)  
  return list_mae

def init_timer():
  start_time = time.monotonic()
  return start_time

def compute_execution_time(start_time:float)->datetime.timedelta:
  end_time = time.monotonic()
  exec_time = datetime.timedelta(seconds=end_time - start_time)
  return exec_time

def log(model, test_nm, exec_time, training=False):
  mode = "generating"
  if training: mode="training"  
  with open(f"trainings/{model}/{test_nm}/LOGS/{mode}_log.txt", mode="w") as f:
    f.write(f"{exec_time}")