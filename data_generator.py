"""
" Generates the synthetic data using a trained model
" TODO: check if experiment path exists
"
"""
import utils as u
import subprocess
import numpy as np

def synth_crnngan(exp_json, n_samples, test_lists):
    """
    Generate data from a C-RNN-GAN model

    Params:
        - exp_json: json file with experiment settings
        - n_samples: number of samples to be generated
        - test_lists: list of tests in the experiment settings

    """
    # currently it generates data using the last trained model
    command = "python models/c_rnn_gan/rnn_gan_compat.py"
    for t in test_lists:             
        exp_json['tests'][t]['sample'] = "True"
        exp_json['tests'][t]['n_synth_sample'] = n_samples
        

        key_parameters = exp_json["tests"][t]
        parameter_lists = ""
        for k in key_parameters:
            parameter_lists += " --{} {}".format(k,exp_json["tests"][t][k])
        command += parameter_lists

        print ("Generating {} samples for test {}".format(n_samples,t))
        subprocess.call(command, shell=True)

def synth_rgan(exp_json, n_samples, rgan_params, test_lists):
    """
    Generate data from RGAN models

    Params:
        - exp_json: json file with experiment settings
        - n_samples: number of samples to be generated
        - test_lists: list of tests in the experiment settings

    """

    # import model
    from models.rgan import model as m

    param_id = rgan_params['param_id']
    path_samples = "experiments/"
    for t in test_lists:
        settings = exp_json['tests'][t]
        identifier = settings['identifier']
        path_parameters = 'experiments/{}/{}/parameters/'.format(exp_json['exp_id'], t)
        path_samples_test = "{}{}/{}/synthdata/".format(path_samples, exp_json['exp_id'], t)
        epochs = settings['num_epochs']
        save_freq = settings['save_freq']
        for ep in range(0, epochs, save_freq):
            try:
                samples = m.sample_trained_model(settings, ep, settings['num_samples'], path_parameters=path_parameters)
                sample_name = "{}_{}_{}".format(exp_json['exp_id'],t,ep)
                save_samples(path_samples_test, sample_name, samples)
            except IOError as e:
                print ("Sample error in test {}, epoch: {}".format(t,ep))
                print ("You are, probably, trying to sample from a non-saved model (at epoch {})".format(ep))
                print ("Ignoring...")
                pass
        
def synth_timegan(exp_json, n_samples, test_lists):

    command = "python models/timegan/generate_samples.py"
    path_to_exp = "experiments/{}".format(exp_json['exp_id'])
    for t in test_lists:
        settings = exp_json['tests'][t]
        path_to_exp_t = path_to_exp+"/"+t
        command += " --dataset {} --path_to_exp {} --n_samples {}".format(
            settings["dataset"],
            path_to_exp_t, n_samples)
        subprocess.call(command, shell=True)

def synth_data(model, exp_file, n_samples=5, rgan_params={'param_id':None}):
    #TODO: se modelo for c-rnn-gan tem que pegar o checkpoint
    exp = u.load_experiment(exp_file)
    test_lists = exp['tests'].keys()

    if model == "crnngan":
        synth_crnngan(exp, n_samples, test_lists)
    elif model == "rgan":
        synth_rgan(exp, n_samples, rgan_params, test_lists)
    elif model == "timegan":
        synth_timegan(exp, n_samples, test_lists)
    else:
        print ("Model {} doesn't exist. The avaliable models are:\n".format(model)+
        "- crnngan\n- rgan\n- timegan\n- \arima")
        exit()

def save_samples(path,sample_name, samples):
    """
    Save generated samples

    Params:
        -path: path where the samples will be saved ('experiments/<exp_name>/<test>/synthdata')
        -sample_name: synth data name (<exp_name>_<test>_<epoch>)
        -samples: generated data
    """


    np.save("{}{}.npy".format(path,sample_name), samples)


# synth_data("crnngan","crnngan_exp1.json", n_samples=3)
# synth_data("rgan","rgan_exp1.json", n_samples=3, rgan_params={'param_id':'test2_1'})
synth_data("timegan", "timegan_exp1.json", n_samples=2)