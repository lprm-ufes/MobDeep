from keras.layers.merge import concatenate
import utils as u
import plotting as p
import numpy as np
import glob
import pandas as pd

from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, GRU
from keras.callbacks import EarlyStopping



def make_rnn_model(units, n_layers=3, n_steps=24, n_features=1, output_units=1, net_type='GRU'):
  """
  Make a Recurrent Neural Network model

  Params:

    - units: rnn units size
    - n_layers: number of layers
    - n_steps: timestep of the data
    - n_features: number of features in data
    - output_units: number of output units imn the RNN
    - net_type: net_type to use (GRU or LSTM)
  
  Returns:

    - model: a RNN model (needs to be compiled)
  """

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

def load_rgan_data(data_dir, data_load_from):
  """
  Load data used in RGAN models.
  RGAN data is organized as a python dict:
  {'pdf': [],
  'labels':[],
  'samples':{
    'train': []
    'test': [],
    'vali': []}}

  Params:

    - datadir: parent folder of dataset
    - data_load_from: specific path for dataset inside 'datadir'

  Returns:

    -
  """
  
  filename = "{}.data.npy".format(data_load_from)
  data = np.load(data_dir+filename,allow_pickle=True)
  
  # get arrays from each key in dict
  train = data.item().get('samples').get('train')
  test  = data.item().get('samples').get('test')
  vali  = data.item().get('samples').get('vali')
  # concatenate splited data
  concatenated_data = np.concatenate([train, test, vali], axis=0)
  
  return concatenated_data



def load_fake_data(exp_json=None, test=""):
    """
    Return the list of fake data names to be loaded using numpy

    Params:

      - exp_json: a json file with the experiment settings
      - test: the current test from the experiment settings file
    
    Returns:
      - listOfFakes: A list of generated files
    """
    # TODO: test this commented line with crnngan
    # the ['traindir'] seems to be useless
    # synth_path = exp_json['tests'][test]['traindir'] + "/synthdata"
    #synth_path = exp_json['tests'][test] + "/synthdata"
    synth_path = "experiments/{}/{}/synthdata/".format(exp_json['exp_id'], test)
    listOfFakes = glob.glob(synth_path+"*.npy")    
    return listOfFakes   
    
def load_real_data(exp_json, model=None):
  """
  Loads the real dataset used in a experiment.

  Params:

      - exp_json: experiment file in .json format
  
  Returns:

      - real_data: a np.array()
  """

  list_tests = exp_json['tests'].keys()
  datadir = exp_json['tests'][list_tests[0]]['datadir']
  if model == 'crnngan':
    filename = exp_json['tests'][list_tests[0]]['filename']
  elif model == 'rgan':
    data_load_from = exp_json['tests'][list_tests[0]]['data_load_from']
    return load_rgan_data(datadir, data_load_from)

  real_data  = np.load(datadir+filename, allow_pickle=True)
  real_data_shp = real_data.shape
  real_data = real_data.reshape(real_data_shp[0]*real_data_shp[1],real_data_shp[-1])
  return real_data

def generate_visualizations(exp_file="", wkday=2):
    """
    TODO: Accept more then one weekday
    """
    exp_json = u.load_experiment(exp_file)
    real_data = load_real_data(exp_json)
    
    # print (exp_json['tests']['t1']['traindir'])
    config_timedelta = {
        'm': 30
    }
    list_dates = p.get_list_dates(data_size=real_data.shape[0],
                year=2020, month=1,day=1, dict_timedelta=config_timedelta)

    df_real = p.get_df(list_dates=list_dates, real_data=real_data, timesteps=48)
    dict_cnt_real = p.get_count(df_real,4,48,column='ts')
    tests = exp_json['tests'].keys()
    for t in tests:
        listOfFakes = load_fake_data(exp_json, '{}'.format(t))
        plots_dir = exp_json['tests'][t]['traindir'] + "/evaluations/plots/"
        for fk in listOfFakes:
            fname, _ = u.os.path.basename(fk).split(".")
            synth = np.load(fk)
            synth_shp = synth.shape
            plots_dir_fk = u.os.path.join(plots_dir,fname)
            try: u.os.mkdir(plots_dir_fk)
            except: pass
            figtt = "{}_{}".format(exp_json['exp_id'],t)
            p.plot_sum_interval(dict_cnt_real=dict_cnt_real, list_dates=list_dates,wkday=4,timestep=48,
                                fake_data=synth.reshape(synth_shp[0]*synth_shp[1], synth_shp[-1]),
                                figtitle=figtt, plots_dir = plots_dir_fk)
        print ("Visualization for test {} saved!.".format(t))

def split_sequence(sequence=None, n_steps=1):
    """
    Devides a sequence into input (X) and output (y)

    Params:

        - sequence: a array
        - n_steps: timestep of the data
    
    Returns: 

        - (X, y): a tuple with the input and output arrays
    """
    shps = sequence.shape
    if len(shps)>2:
        sequence = sequence.reshape(shps[0]*shps[1], shps[-1])

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


def estocastic_residuals(exp_json):
  #TODO: implement residual for ARIMA
  raise NotImplementedError


def deep_residuals(exp_json, real_data, nn_params):
  """
  Get the residuals for deep learning based models

  Params:

    - exp_json: the experiment json file
    - real_data: the real data
    - nn_params: the parameters to be used in the model definition
  """
  exp_id = exp_json["exp_id"]
  # FIXME: exp_file must be passed as a parameter
  fname = exp_file.split(".")[0]
  # FIXME: change modelName to modelPath
  model_path = "experiments/{}/{}".format(exp_id,exp_id)
  
  # try to load a saved rnn model from experiments/<exp_id>/<exp_id>
  try:
    rnn_model = load_model(model_path)
    X, y = split_sequence(real_data,n_steps=nn_params['n_steps'])
    print ("X: ", X.shape)
    print ("Model loaded.")
  except:
    print ("Model not found. Saving one at {}.".format(model_path))
    
    # stop training when model it is not improving anymore
    earlystop = EarlyStopping(monitor='loss')
    # make model
    rnn_model = make_rnn_model(24,n_layers=2, n_steps=nn_params['n_steps'],n_features=nn_params['nfeatures'],output_units=nn_params['nfeatures'],net_type='LSTM')
    # if executing on python2
    if (u.sys.version_info[0]==2):
      opt = Adam(lr=5e-4)
    else:
      opt = Adam(learning_rate=5e-4)
    
    # compile model
    rnn_model.compile(optimizer=opt,loss='mse')
    # get input and output for real data
    X, y = split_sequence(real_data,n_steps=nn_params['n_steps'])
    # train the model on real data
    rnn_model.fit(X, y, batch_size=nn_params['bs'], epochs=nn_params['e'], callbacks=[earlystop],verbose=nn_params['v'])
    # saves the model
    rnn_model.save(model_path)

    print ("Model trained and saved.")
  
  tests = exp_json['tests'].keys()


  y_pred = rnn_model.predict(X)
  dict_resid_tests = {}
  # compute the residuals for each test
  for t in tests:
    # dict_resid = {}
    means = list()
    stds  = list()
    listOfFakes = load_fake_data(exp_json, '{}'.format(t))
    # synth_dir = exp_json['tests'][t]['traindir'] + "/evaluations/residuals"
    
    for fk in listOfFakes:
      synth = np.load(fk)
      # TODO: pass timesteps through args
      X, y = split_sequence(synth,n_steps=nn_params['n_steps'])
      e_r = y - y_pred
      means.append(e_r)
      stds.append(e_r)
    
    mean_rs = np.mean(means)
    std_rs  = np.std(stds)
    dict_resid_tests[t] = [mean_rs, std_rs]
  
  df_resid_tests = pd.DataFrame(dict_resid_tests, index=['mean','std'])
  resid_dir = 'experiments/{}/residuals/'.format(exp_json['exp_id'])
  try: u.os.makedirs(resid_dir)
  except: pass
  df_resid_tests.to_csv(resid_dir+'df_resid_tests.csv')

  return df_resid_tests

def get_best_test(df_resid, treshold=0.5):
  """
  Returns the test name that produces the best 

  TODO: Update to return the best model insted of test name

  Params:

    - df_resid: dataframe with model residuals
    - treshold: minimum value acceptebla for the model std
  """
  best = df_resid.idxmin(axis=1)['mean']
  std_best = df_resid[best]['std']
  message = ""
  if (std_best <= treshold):
    message = best
  else:
    message = "Model std is larger than the {} treshold".format(treshold)
  return message

def generate_residuals(exp_file=None, modeltype = 'arima', model=None, nn_params={'bs':200, 'e':50, 'v':False}):
  """
  Method to compute the models residuals

  Params:

    - exp_file: the experiment json file name
    - modelname: the model to calculate the residuals
    - nn_params: deep learning parameters. Only used in the deep leared-based residuals
  """
  # TODO: Calculate best model for each saved model by each test
  # TODO: make residual for ARIMA


  if (exp_file == None):
    print ("Please, set a experiment file")
    exit()

  exp_json = u.load_experiment(exp_file)
  real_data = load_real_data(exp_json, model)
  # TODO
  df_resid = None
  if (modeltype == 'arima'):
    df_resid = estocastic_residuals(exp_json)
  elif(modeltype=='gans'):
    df_resid = deep_residuals(exp_json, real_data, nn_params)
  print ("Residuals saved.")  


 
exp_file = "crnngan_exp1.json"
exp_rgan_file="rgan_exp1.json"
# generate_visualizations(exp_file)
# TODO: get nnparams from exp_file
# model = 'rgan'
# generate_residuals(exp_file=exp_rgan_file, modeltype='gans',
#                   model=model, nn_params={'bs':120, 'e':1, 
#                   'v':True, 'nfeatures':1, 'n_steps':48})

df = pd.read_csv("experiments/rgan_exp1/residuals/df_resid_tests.csv")

print(get_best_test(df, 0.16))