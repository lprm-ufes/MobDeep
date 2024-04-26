#TODO: read parameters from args
import numpy as np
import utils as u
import subprocess

def train_arima():
    raise NotImplementedError

def train_crnngan(exp_settings):
    """
    Train crnngan models

    Params:

        - exp_settings: trainings settings for the models
    """
    command = "python models/c_rnn_gan/rnn_gan_compat.py"        
    parameter_lists = ""
    test_list = exp_settings["tests"].keys()
    for t in test_list:
        key_parameters = exp_settings["tests"][t]
        for k in key_parameters:
            parameter_lists += " --{} {}".format(k,exp_settings["tests"][t][k])
        command_t = command + parameter_lists
        subprocess.call(command_t, shell=True)

def train_rgan(exp_settings):
    """
    Train rgan models

    Params:

        - exp_settings: trainings settings for the models
    """
    command = "python models/rgan/experiment.py"
    parameter_lists = ""
    test_list = exp_settings["tests"].keys()
    for t in test_list:
        key_parameters = exp_settings["tests"][t]
        for k in key_parameters:
            parameter_lists += " --{} {}".format(k,exp_settings["tests"][t][k])
        parameter_lists +=" --exp_id {}".format(exp_settings['exp_id'])
        parameter_lists +=" --test_id {}".format(t)
        command_t = command + parameter_lists
        subprocess.call(command_t, shell=True)

def train_timegan():
    raise NotImplementedError

def train_models(model, exp_settings):
    """
    Train a model

    Params:

        - model: model used for training
        - exp_settings: experiment settings for the model
    """
    
    if model == "arima":
        train_arima()
    elif model == "crnngan":
        train_crnngan(exp_settings)        
    elif model == "rgan":
        train_rgan(exp_settings)
    elif model == "timegan":
        train_timegan(exp_settings)
    else:
        print ("Model {} doesn't exist. The avaliable models are:\n".format(model)+
        "- crnngan\n- rgan\n- timegan\n- \arima")
        exit()

""""
Tests:
    -TODO: rename exp files
"""
def test_crnngan():
    exp_file = "crnngan_exp1.json"
    model = "crnngan"
    exp_settings = u.load_experiment(exp_file)
    exp_id = exp_settings['exp_id']
    u.creat_directories(exp_id, exp_settings)
    train_models(model,exp_settings)

def test_rgan():
    model = "rgan"
    exp_file = "rgan_exp1.json"
    exp_settings = u.load_experiment(exp_file)
    exp_id = exp_settings['exp_id']
    u.creat_directories(exp_id, exp_settings)
    train_models(model,exp_settings)

def test_timegan():
    model = "timegan"
    exp_settings = u.load_experiment(model)
    exp_id = exp_settings['exp_id']
    print (exp_id)
    print (exp_settings['tests'].keys())
    u.creat_directories(exp_id, exp_settings)

def test_arima():
    model = "arima"
    exp_settings = u.load_experiment(model)
    exp_id = exp_settings['exp_id']
    print (exp_id)
    print (exp_settings['tests'].keys())
    #u.creat_directories(exp_id, exp_settings)

model = "crnngan"
   

# dict_settings = load_experiment(model)
# arguments = ""
# for k in dict_settings:
#     temp = "{} {}".format(k, dict_settings[k])
#     arguments =  arguments + "{} ".format(temp)

#os.system("python models/c_rnn_gan/rnn_gan_compat.py {}".format(arguments))
experiment_id = "exp1"
tests = ["t1","t2", "t3"]
#u.creat_directories(experiment_id, tests)
#test_crnngan()
test_rgan()