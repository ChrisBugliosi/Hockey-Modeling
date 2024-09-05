import pandas as pd
import numpy as np
from patsy.highlevel import dmatrices
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from os import path

DATA_DIR = '/Users/chrisbugs/Downloads/code-hockey-files-0.2.1 2/data'

df = pd.read_csv(path.join(DATA_DIR, 'shots.csv'))
df = df.loc[df['shot_type'].notnull()]

cont_vars = ['dist', 'st_x', 'st_y', 'period_time_remaining', 'empty']
cat_vars = ['pos', 'hand', 'period']

# replace periods 1, 2, 3 ... with P1, P2, P3 ...
# this is so that when we turn them into dummy variables the column names are
# P1, P2, ... and not just 1, 2, which can cause issues
df['period'] = 'P' + df['period'].astype(str)

df_cat = pd.concat([pd.get_dummies(df[x]) for x in cat_vars], axis=1)

df_all = pd.concat([df[cont_vars], df_cat], axis=1)
df_all['shot_type'] = df['shot_type']
df_all.sample(10)

yvar = 'shot_type'
xvars = cont_vars + list(df_cat.columns)
xvars

train, test = train_test_split(df_all, test_size=0.20)

model = RandomForestClassifier(n_estimators=100)
model.fit(train[xvars], train[yvar])

test['shot_type_hat'] = model.predict(test[xvars])
test['correct'] = (test['shot_type_hat'] == test['shot_type'])
print(test['correct'].mean())

print(model.predict_proba(test[xvars]))

probs = DataFrame(model.predict_proba(test[xvars]),
                  index=test.index,
                  columns=model.classes_)
print(probs.head(10))
# probs.columns = ['pbhand', 'pdeflct', 'pslap', 'psnap', 'ptip', 'pwrap', 'pwrist']

results = pd.concat([test[['shot_type', 'shot_type_hat',
    'correct']], probs], axis=1)


results.groupby('shot_type')[['correct', 'backhand', 'deflected', 'slap',
    'snap', 'tip-in', 'wrap-around', 'wrist']].mean().round(2)

# cross validation
model = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(model, df_all[xvars], df_all[yvar], cv=10)

scores
print(scores.mean())

# feature importance
model = RandomForestClassifier(n_estimators=100)
model.fit(df_all[xvars], df_all[yvar])  # running model fitting on entire dataset
print(Series(model.feature_importances_, xvars).sort_values(ascending=False))

"""
TO-DO: Check this out!!!!
"""

# logReg model
df['ln_dist'] = np.log(df['dist'].apply(lambda x: max(x, 0.5)))
df['goal'] = df['goal'].astype(int)
y, X = dmatrices('goal ~ dist', df)

# Model created
model = LogisticRegression()

# Cross-validation tested
scores = cross_val_score(model, X, y.ravel(), cv=10)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model.fit(X_train, y_train.ravel())

# Predict on test data
y_pred = model.predict(X_test)  # Make predictions
y_pred_proba = model.predict_proba(X_test)  # Get prediction probabilities

# Ensure y_test is a DataFrame for concatenation
y_test_df = pd.DataFrame(y_test, columns=['goal']).reset_index(drop=True)

# Create DataFrame with probabilities
log_probs_df = pd.DataFrame(y_pred_proba, columns=['prob_not_goal', 'prob_goal'])
log_results = pd.concat([y_test_df, log_probs_df], axis=1)
log_results['predicted'] = y_pred

# Add a 'correct' column
log_results['correct'] = log_results['goal'] == log_results['predicted']

# How many the model got correct
print(log_results['correct'].sum())
# How many there were
print(log_results.shape[0])
# Success rate
print(log_results['correct'].sum()/log_results.shape[0])
