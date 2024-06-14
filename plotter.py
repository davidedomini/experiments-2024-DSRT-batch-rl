import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import glob


def filter_line(line):
    return ('#' not in line) and ('NaN' not in line)


def is_header(line):
    return 'time' in line


def clean_alchemist_csv(path, output_dir):
    files = glob.glob(f'{path}/*.csv')
    for f in files: 
        cleaned = []
        with open(f, 'r+') as file:
            for line in file:
                if is_header(line):
                    cleaned.append(line[2:])
                elif filter_line(line):
                    cleaned.append(line)

        with open(f'{output_dir}{f.split('/')[-1]}', 'w') as file:
            file.writelines(cleaned)


def plot_metric(data, metric, round, experiment, yname):
    plt.figure(figsize=(12, 8))
    colors_v = sns.color_palette("colorblind", 10) 
    ax = sns.lineplot(data=data, x='time', y=metric, color=colors_v[0])
    plt.xlabel('time')
    plt.ylabel(yname)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.title(f'Global round {round}')
    name = yname.replace(' ', '-').lower()
    plt.savefig(f'charts/{experiment}-{name}-{round}.pdf', dpi=500)
    plt.close()


def to_label(experiment):
    if '1' in experiment: 
        return 'Parallelism 1'
    elif '2' in experiment: 
        return 'Parallelism 2'
    elif '4' in experiment: 
        return 'Parallelism 4'
    elif '8' in experiment: 
        return 'Parallelism 8'


def plot_metric_all(data, metric, round, yname):
    plt.figure(figsize=(12, 8))
    colors_v = sns.color_palette("colorblind", 4) 
    for i, d in enumerate(data):
        l = to_label(d["Experiment"].iloc[0])
        ax = sns.lineplot(data=d, x='time', label=l, y=metric, color=colors_v[i])
    plt.xlabel('Time')
    plt.ylabel(yname)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.legend()
    # sns.move_legend(ax, 'upper right')
    plt.title(f'Global round {round}')
    name = yname.replace(' ', '-').replace('$', '').lower()
    plt.savefig(f'charts/all-{name}-{round}.pdf', dpi=500)
    plt.close()

def line_plot(data, metric, labels):
    plt.figure(figsize=(12, 8))
    colors_v = sns.color_palette("colorblind", 4) 
    for i, exp in enumerate(labels):
        l = to_label(exp)
        ax = sns.lineplot(data=data, x='Global Round', label=l, y=exp, color=colors_v[i])
    plt.xlabel('Global Round')
    plt.ylabel(metric)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.legend()
    # sns.move_legend(ax, 'upper right')
    plt.title(f'Global round {round}')
    plt.savefig(f'charts/global-{metric}-{round}.pdf', dpi=500)
    plt.close()

def avg_std_dataframes(dfs):
    # Calculate the mean DataFrame
    mean_df = sum(dfs) / len(dfs)

    # Calculate the standard deviation DataFrame
    std_df = pd.concat(dfs).groupby(level=0).std()
    minus_deviation = mean_df - std_df
    plus_deviation = mean_df + std_df
    minus_deviation['time'] = mean_df['time']
    plus_deviation['time'] = mean_df['time']
    ## stack the dataframes with the mean in one dataframe
    return pd.concat([mean_df, minus_deviation, plus_deviation])


if __name__ == '__main__':

    matplotlib.rcParams.update({'axes.titlesize': 20})
    matplotlib.rcParams.update({'axes.labelsize': 18})
    matplotlib.rcParams.update({'xtick.labelsize': 15})
    matplotlib.rcParams.update({'ytick.labelsize': 15})
    plt.rcParams.update({"text.usetex": True})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

    input_dir = 'data/'
    clean_data_dir = 'data-cleaned/'
    charts_dir = 'charts/'
    experiments = ['experiment-par-1', 'experiment-par-2', 'experiment-par-4', 'experiment-par-8']
    clean = True
    global_rounds = 9


    if clean:
        clean_alchemist_csv(input_dir, clean_data_dir)


    # for experiment in experiments:
    #     for round in range(1, global_rounds+1):
    #         files = glob.glob(f'{clean_data_dir}{experiment}*globalRound-{round}.csv')        
    #         dataframes = []
    #         for f in files:
    #             df = pd.read_csv(f, sep = ' ')
    #             dataframes.append(df)
            
    #         mean_df = sum(dataframes)/len(dataframes)
    #         mean_and_std = avg_std_dataframes(dataframes)
    #         plot_metric(mean_and_std, 'Reward[mean]', round, experiment, 'Reward')
    #         plot_metric(mean_and_std, 'MeanDistance[mean]', round, experiment, 'Distance from neighboors')


    metrics = ['Reward[mean]', 'MeanDistance[mean]']

    for metric in metrics:
        data_xglobal = pd.DataFrame()
        data_xglobal['Global Round'] = [x for x in range(0, global_rounds)]
        for experiment in experiments:
            values = []
            for round in range(1, global_rounds+1):
                files = glob.glob(f'{clean_data_dir}{experiment}*globalRound-{round}.csv')               
                dataframes = []
                for f in files:
                    df = pd.read_csv(f, sep = ' ')
                    dataframes.append(df)
                mean_df = sum(dataframes)/len(dataframes)
                values.append(mean_df[metric].iloc[-1])
            data_xglobal[experiment] = values
        line_plot(data_xglobal, metric, experiments)


    for round in range(1, global_rounds+1):
        dataframes = []
        print(f'Generating charts for round {round}')
        for experiment in experiments:
            files = glob.glob(f'{clean_data_dir}{experiment}*globalRound-{round}.csv')
            dfs_epx = []
            for f in files:
                df = pd.read_csv(f, sep = ' ')
                dfs_epx.append(df)
            mean_and_std = avg_std_dataframes(dfs_epx)
            mean_and_std["Experiment"] = experiment
            dataframes.append(mean_and_std)

        plot_metric_all(dataframes, 'MeanDistance[mean]', round, 'Distance from neighboors')
        plot_metric_all(dataframes, 'Reward[mean]', round, 'Reward')
