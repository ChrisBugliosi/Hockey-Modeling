import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from os import path


DATA_DIR = '/Users/chrisbugs/Downloads/code-hockey-files-0.2.1 2/data'


dfpg = pd.read_csv(path.join(DATA_DIR, 'player_games.csv'))
dfs = pd.read_csv(path.join(DATA_DIR, 'shots.csv'))

dfpg = pd.get_dummies(dfpg, columns=['pos'])

"""
GENERIC MODEL IMPLEMENTATION
"""

def rfr_creator(x_vars, y_vars, df):
       # Train test split
       train, test = train_test_split(df, test_size=0.20)

       # Create & train model
       model = RandomForestRegressor(n_estimators=250)
       model.fit(train[x_vars], train[y_vars])

       # Apply model
       test['goals_hat'] = model.predict(test[x_vars])
       #test['goals_prob'] = model.predict_proba(test[x_vars])
       test['correct'] = test['goals_hat'] == test['goals']

       # Cross validation
       model = RandomForestRegressor(n_estimators=100)
       scores = cross_val_score(model, df[x_vars], df[y_vars], cv=10)

       # Returns
       return model, test, scores

def rfr_w_prediction(x_vars, y_vars, df, new_data):
       # Allign new data
       new_data = new_data[x_vars]

       # Create the model
       model = RandomForestRegressor(n_estimators=250)
       model.fit(df[x_vars], df[y_vars])

       # Feature importance
       features_imp = Series(model.feature_importances_, x_vars).sort_values(ascending=False)

       # Predictions
       predictions = DataFrame(model.predict(new_data))

       # Cross validation
       model = RandomForestRegressor(n_estimators=100)
       scores = cross_val_score(model, df[x_vars], df[y_vars], cv=10)

       # Returns
       return model, features_imp, scores, predictions

"""
RANDOM FOREST REGRESSOR PLAYER-GAME TO GOALS
"""

# X and Y vars
x_vars = ['shots', 'assists', 'hits', 'time_ice', 'pen_min', 'pp_goals', 'pp_assists',
       'fo_wins', 'fo', 'takeaways', 'giveaways', 'goals_sh', 'assists_sh',
       'blocks', 'plus_minus', 'time_ice_even', 'time_ice_sh', 'time_ice_pp']
y_vars = 'goals'

train, test = train_test_split(dfpg, test_size=0.20)

pg_model, pg_features, pg_scores, pg_predictions = rfr_w_prediction(x_vars, y_vars, train, test)

print(f"{pg_features} \n\n {pg_scores.mean()} \n\n {pg_predictions[['name','goals']].head()}")