import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('computingTime.csv', sep=',')
print(df)
df["computingWithoutParallelism"] = df["parallelism"] * df["computingTime"][0]
ax = df.plot(x='parallelism', y='computingWithoutParallelism', kind='bar', color='red')
ax = df.plot(x='parallelism', y='computingTime', kind='bar', ax=ax)

plt.show()
