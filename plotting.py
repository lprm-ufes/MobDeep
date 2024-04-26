import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# TODO: document the functions

def get_time_delta(dict_timedelta):
    current_keys = ['y','M','wk','d','h', 'm', 's']
    dict_time = {}

    # set the timedelta to creat the list of dates
    for k in current_keys:
        if k not in dict_timedelta:
            dict_time[k] = 0
        else:
            dict_time[k] = dict_timedelta[k]
    

    tdelta = datetime.timedelta(
        hours   = dict_time['h'],
        minutes = dict_time['m'],
        seconds = dict_time['s'])
    return tdelta

def get_list_dates(data_size=None, year=None, month=None, day=1, dict_timedelta={'h':1}):
    """
    Creates a list of dates for the dataset. It depends on the data size and data frequency

    Params:
        
        - data_size: number of samples in data
        - y: year
        - m: month
        - d: day
        -
    """
    date = datetime.datetime(year,month,day,0)
    tdelta = get_time_delta(dict_timedelta)


    list_dates = []
    for i in range(data_size):
        list_dates.append(date)
        date = date + tdelta
    
    return list_dates

def get_count(df, wkday, timestep,column='hr'):
    """
    Sums the variables in the data by each interval
    TODO: accetp weekday as list of days
    Params:

        - df: A dataset with the date, hour, week, minutes and variables columns
        - wkday: weekday to filter the data
        - timestep: the data timestep
        - column: column in the data that will be used for computing the sums
    """
    df_wkday = df[df['wk'] == wkday]
    dict_cnt_column = dict()       
    for col in df.columns:
        if "cnt" in col:
            dict_cnt_column[col] = [df_wkday[df_wkday[column]==i][col].sum() for i in range(timestep)]
    
    return dict_cnt_column

def add_week_hr(df, timesteps=24):
    """
    TODO: test when hour = 2

    Add additional columns (hour, week and minutes) to the dataframe

    Params

        - df: a pandas dataframe. Must have a column named 'date' that is a datetime object
    """

    # TODO: get column based on parameters
    df['hr'] = df.apply(lambda r: r.date.hour, axis=1)
    df['wk'] = df.apply(lambda r: r.date.weekday(), axis=1)
    df['min'] = df.apply(lambda r: r.date.minute, axis=1)

    return df

def get_df(list_dates=None, real_data=None, timesteps=24):
    """
    Returns a pandas dataframe the columns to be used in the visualizations

    Params:

        - list_dates: list of dates
        - real_data: the real data that will be used to create the pandas dataframe
    """
    
    dict_real = {'date':list_dates}

    # creates a pandas column for each variable in the data
    for i in range(real_data.shape[1]):
        dict_real['cnt_{}'.format(i)] = real_data[:,i]


    df_real = pd.DataFrame(dict_real)
    df_real = add_week_hr(df_real)
    ts = [t for t in range(timesteps)]
    df_real['ts'] = ts * (real_data.shape[0]//timesteps)

    return df_real

def plot_sum_interval(dict_cnt_real = {}, fake_data = None, list_dates=None, figtitle="",
                    wkday=0, plots_dir="", timestep=0, fmt="png"):
    """

    """

    df_fake = get_df(list_dates,fake_data)
    # TODO: pass column, xlabel as arg
    # dict_cnt_real = get_count(df_real,2,24,column='hr')
    dict_cnt_fake = get_count(df_fake,wkday,timestep,column='ts')

    for k in dict_cnt_real:
        cnt_real = dict_cnt_real[k]
        cnt_fake = dict_cnt_fake[k]

        plt.plot(cnt_real,label='{}_real'.format(k))
        plt.plot(cnt_fake,label='{}_fake'.format(k))
        plt.xlabel("hr")
        plt.legend()
        plt.title("{}".format(figtitle))
        # plots_dir should exist
        plt.savefig("{}/sum_int_{}_{}.{}".format(plots_dir, figtitle, k, fmt))
        plt.clf()
        plt.close()