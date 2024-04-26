import os
import sys
import json as j

error_messages = {
    "path_exists": "Directory already exists.  Continuing...",
}

if (sys.version_info[0] == 2):
    import errno
    class FileExistsError(OSError):
        def __init__(self, msg):
            super(FileExistsError, self).__init__(errno.EEXIST, msg)



def get_list_tests(exp_settings):
    """
    Get the list of tests for the experiment
    """

    list_tests = [k for k in exp_settings['tests'].keys()]
    return list_tests

def load_experiment(exp_file_name):
    #TODO: try to load a file that doesn't exist
    """
    Loads a experiment file
    """
    
    f = open("experiments/"+exp_file_name)
    experiment_file = j.load(f)
    return experiment_file

def creat_directories(exp_id, exp_settings):
    """
    Create the directories for the experiment
    
    Pattern:
    exp_id
        - models
        - synthdata
        - evaluations
            - plots
            - residuals
    """

    path_experiment = os.path.join("experiments","{}".format(exp_id))
    try:
        os.makedirs(path_experiment)
        print ("Experiment directory created")
    except FileExistsError as e :
        print (error_messages["path_exists"])
    except OSError as e:
        print (error_messages["path_exists"])
    list_tests = get_list_tests(exp_settings)
    for t in list_tests:
        path_temp_models = os.path.join(path_experiment,"{}/models".format(t)) # path to models
        path_temp_synth = os.path.join(path_experiment,"{}/synthdata".format(t)) # path to generated data
        path_temp_eval = os.path.join(path_experiment,"{}/evaluations".format(t)) # path to evaluations
        path_temp_eval_plots = os.path.join(path_temp_eval,"plots") # path to evaluations plots
        path_temp_eval_resid = os.path.join(path_temp_eval,"residuals") # path to evaluations residuals

        try:
            os.makedirs(path_temp_models)
            os.makedirs(path_temp_synth)
            os.makedirs(path_temp_eval)
            os.makedirs(path_temp_eval_plots)
            os.makedirs(path_temp_eval_resid)
            print("Sub-directories created")
        except FileExistsError as e:
            print (error_messages["path_exists"])
        except OSError as e:
            print (error_messages["path_exists"])