import pandas as pd
import numpy as np
import seaborn as sns
from patsy.highlevel import dmatrices
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.formula.api as smf
from os import path

DATA_DIR = '/Users/chrisbugs/Downloads/code-hockey-files-0.2.1 2/data'

dfpg = pd.read_csv(path.join(DATA_DIR, 'player_games.csv'))

model = smf.ols(formula='goals ~ time_ice + time_ice_pp + hits + C(pos) + shots', data=dfpg)
results = model.fit()

print(results.summary2())

dfpg['goals_hat'] = results.predict(dfpg)

# X and Y vars
x_vars = ['shots', 'goals', 'assists', 'hits', 'time_ice', 'pen_min', 'pp_goals', 'pp_assists',
       'fo_wins', 'fo', 'takeaways', 'giveaways', 'goals_sh', 'assists_sh',
       'blocks', 'plus_minus', 'time_ice_even', 'time_ice_sh', 'time_ice_pp']
y_vars = 'pos'

# Train test split
train, test = train_test_split(dfpg, test_size=0.20)

# Create and train model
model = RandomForestClassifier(n_estimators=250)
results = model.fit(train[x_vars], train[y_vars])

# Cross validate
cv = cross_val_score(model, dfpg[x_vars], dfpg[y_vars], cv=10)
print(cv.mean())

# Feature importance
print(Series(model.feature_importances_, x_vars).sort_values(ascending=False))

# Accuracy check
test['pos_hat'] = model.predict(test[x_vars])
test['correct'] = (test['pos_hat'] == test['pos'])
print(test['correct'].mean())