import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def render_single_round(df):
    fig, ax = plt.subplots()


data_path = 'data/positions-8_seed-4.0_spacing-1_error-0.5_globalRound-9.csv'

df = pd.read_csv(data_path, skiprows=11, header=None, sep=' ', skipfooter=3)
take_x = list(range(1, 101))
take_y = list(range(101, 201))
Xs = df[take_x]
Ys = df[take_y]
alphas = range(0, 100)
alphas = [x/1000 for x in alphas]
# Define the colormap
colors = sns.color_palette("husl", 100)

for i in range(1, 100):
    plt.scatter(Xs[0:198][[i]], Ys[0:198][[i+100]], s=2, alpha=0.02, c=colors[i])

for i in range(1, 100):
    ## dark color of i
    dark = colors[i]
    dark = [x/1.5 for x in dark]
    plt.scatter(Xs[198:199][[i]], Ys[198:199][[i+100]], s=20, alpha=1, c=colors[i], edgecolors=dark, linewidths=1)
#for i in range(0, 10):
#    plt.scatter(Xs.iloc[i,j], Ys.iloc[i,j], s=2, c=colorMap[j])
plt.show()
