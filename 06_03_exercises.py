import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os import path

DATA_DIR = '/Users/chrisbugs/Downloads/code-hockey-files-0.2.1 2/data'

dfpg = pd.read_csv(path.join(DATA_DIR, 'player_games.csv'))

# 6.1

g1 = (sns.FacetGrid(dfpg, hue='pos')
     .map(sns.kdeplot,'time_ice_pp', fill=True)
     .add_legend())

g2 = (sns.FacetGrid(dfpg, col='pos', col_wrap=2)
     .map(sns.kdeplot,'time_ice_pp', fill=True)
     .add_legend())

g1.figure.subplots_adjust(top=0.9)
g1.figure.suptitle('Distribution of Power-Play Time by Pos')

g2.figure.subplots_adjust(top=0.9)
g2.figure.suptitle('Distribution of Power-Play Time by Pos V2')

g3 = (sns.FacetGrid(dfpg, col='team', col_wrap=8)
      .map(sns.kdeplot,'time_ice_pp', fill=True)
      .add_legend())

g3.figure.subplots_adjust(top=0.9)
g3.figure.suptitle('Distribution of Power-Play Time by Team')

#6.2

g4 = sns.relplot(data=dfpg, x='time_ice_pp', y='time_ice_sh')

g4.figure.subplots_adjust(top=0.9)
g4.figure.suptitle('Relationship Between PP Ice-Time and SH Ice-Time')

dfpg['jtime_ice_sh'] = np.random.uniform(dfpg['time_ice_sh'] - 0.5, dfpg['time_ice_sh'] + 0.5)
dfpg['jtime_ice_pp'] = np.random.uniform(dfpg['time_ice_pp'] - 0.5, dfpg['time_ice_pp'] + 0.5)

g5 = sns.relplot(data=dfpg, x='jtime_ice_pp', y='jtime_ice_sh')

g5.figure.subplots_adjust(top=0.9)
g5.figure.suptitle('Relationship Between PP Ice-Time and SH Ice-Time (Jittered)')

print(dfpg[['time_ice_sh', 'time_ice_pp']].corr())

plt.show()