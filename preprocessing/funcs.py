import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def load_data(recording, data_dir, sep, verbose):
    '''
    Loads csv file, reads to a pandas dataframe and returns the dataframe
    Paramerters:
        recording   = recording being loaded
        data_dir    = root directory in which the data is stored (a sub directory for each recording)
        sep         = joiner used for direcotries e.g. '/' or '\\'
        verbose     = True or False
    Returns:
        pandas DataFrame
    '''
    if verbose:
        print('\n\n\nLoading data:\t{}'.format(recording))
    file = sep.join([data_dir, recording, recording]) + '.csv'
    return pd.read_csv(file, index_col=0)


def manipulate_df(df):
    '''
    Manipulates a pandas dataframe into a form suitable for pivoting
    Paramerters:
        df          = pandas DataFrame
    Returns:
        pandas DataFrame
    '''
    df['spike'] = 1
    df['time'] = pd.to_timedelta(df['time'], unit='s')
    return df


def create_ts(df, rolling, resample_period):
    '''
    Pivots a pandas dataframe into one suited for time series analysis (one column per neuron, one row per timepoint)
    Paramerters:
        df               = pandas DataFrame
        rolling          = integer: rolling window over which mean will be calculated. Defaults to None.
        resample_period  = size of time bin
    Returns:
        pandas DataFrame
    '''
    df = df.pivot_table(index='time',
                        columns='spike_cluster',
                        values='spike',
                        aggfunc='count')
    df = df.resample('s').count()
    if rolling:
        df = df.rolling(rolling).mean()
    return df


def calculate_condition_statistics(df):
    '''
    Takes a tidy pandas DataFrame containing spike information in time series.
    Returns mean and standard deviation of firing rate of each neuron by over a specified condition by binning
    Parameters:
        df                  = pandas DataFrame on which statistics will be calculated
        condition           = conditoon over which . Must be a value in the 'condition' column of df
        resample_period     = size of time bin e.g. 'min', 'sec', '10sec' etc
    Returns:
        condition_means     = mean calculated over the interval
        condition_stds      = standard diviation calculated over the interval
        condition_sorted    = sorted index of means
    '''
    condition_means = df.transpose().mean(axis=0)
    condition_stds = df.transpose().std(axis=0)
    condition_sorted = condition_means.sort_values()
    return condition_means, condition_stds, condition_sorted


def normalise(df, method, condition_means, condition_stds, condition_sorted, verbose):
    '''
    Normalises a pandas DataFrame with either percentage baseline mean or zscore of baseline mean
    Parmameters:
        df                      = pandas DataFrame
        method                  = normalisation to use 'zscore' or 'percentage'
        condition_means         = means to which normalisation occurs
        condition_stds          = stds to which normalisation occurs
        condition_sorted        = sorted index of means
        verbose                 = True or False
    Returns:
        normalised pandas DataFrame
    '''
    if verbose:
        print('Normalising...')

    if method == 'zscore':
        def f(col):
            return (col.subtract(condition_means)).divide(condition_stds)
    elif method == 'percent':
        def f(col):
            return col.divide(condition_means) * 100
    elif not method:
        def f(col):
            return col
    df = df.transpose().apply(f)
    return df.reindex(condition_sorted.index)


def gen_fig_path(fig_dir, recording, sep):
    '''
    Generate absolute path to a png figure
    Parameters:
        fig_dir     = root figure folder
        recording   = recording to which the figure corresponds
        sep         = directory seperator for operating system e.g. '/'
    Returns:
        string of absolute path to figure
    '''
    return sep.join([fig_dir, recording]) + '.png'


def plot_heat(df, dpi, vmin, vmax, method, recording, fig_dir, sep, verbose):
    '''
    Plots a heat map of neuronal activity with each row and colour corresponding to one neuron and with time along to x axis
    Parameters:
        df          = pandas DataFrame
        dpi         = resolution of the image e.g. 300
        vmin        = bottom cutt off for colourbar
        vmax        = top cutt off for colourbar
        method      = method previously used for normalisation
        recording   = name of the recording
        fig_dir     = name of the root figure directory
        sep         = os seperator e.g. '/'
        verbose     = True or False
    TODO:
        create line at which conditions change
    '''
    if verbose:
        print('Plotting Heatmap...')
    f, a = plt.subplots(figsize=(19, 9))
    recording_len = df.transpose().index.max().seconds
    x_tick_pos = round(recording_len / 4)
    sns.heatmap(data=df, cmap='coolwarm', vmin=vmin,
                vmax=vmax, xticklabels=x_tick_pos, cbar_kws={'label': f'{method} Baseline mean'},
                ax=a)
    a.set_ylabel('Neuron Number\nSorting by baseline firing rate in ascending order (slowest on top)')
    a.set_xlabel('Time (min)')
    a.set_title(f'Firing Rate Normalised by {method}')
    a.set_xticklabels(list(map(lambda num:
                               str(round(recording_len / 4 / 60 * num, -1)),
                               [0, 1, 2, 3])))
    plt.savefig(gen_fig_path(fig_dir=fig_dir, recording=recording,
                             sep=sep), dpi=600)
