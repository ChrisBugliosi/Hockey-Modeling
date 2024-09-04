import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import zscore
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.image as mpimg
import sqlite3
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from os import path

# Input data
df_raw = pd.read_csv(path.join('/Users/chrisbugs/Downloads/skaters.csv'))
df_goalies = pd.read_csv(path.join('/Users/chrisbugs/Downloads/goalies.csv'))

# Shorten dataframes
df_shortened = df_raw[['playerId', 'season', 'name', 'position', 'team', 'games_played', 'situation', 'I_F_xGoals',
                       'I_F_xOnGoal', 'I_F_flurryAdjustedxGoals', 'I_F_shotsOnGoal', 'I_F_shotAttempts', 'I_F_goals',
                       'I_F_savedUnblockedShotAttempts', 'I_F_highDangerShots', 'I_F_mediumDangerShots',
                       'I_F_lowDangerShots', 'I_F_highDangerxGoals', 'I_F_mediumDangerxGoals', 'I_F_lowDangerxGoals',
                       'I_F_highDangerGoals', 'I_F_mediumDangerGoals', 'I_F_lowDangerGoals', 'icetime']].copy()

df_goalies_shortened = df_goalies[['playerId', 'season', 'name', 'team', 'games_played', 'situation', 'ongoal',
                                   'xGoals', 'goals', 'flurryAdjustedxGoals', 'lowDangerxGoals', 'mediumDangerxGoals',
                                   'highDangerxGoals', 'lowDangerGoals', 'mediumDangerGoals', 'highDangerGoals',
                                   'icetime']].copy()

# TODO: MAYBE SET MINIMUM TO BE IN TOTAL ICETIME NOT GAMES PLAYED
# Min 25 GP for players
df_player = df_shortened[df_shortened['games_played'] >= 25].copy()

# Min 15 GP for goalies
df_goalie = df_goalies_shortened[df_goalies_shortened['games_played'] >= 25].copy()

# Only keep 'all' situations
df_goalie = df_goalie[df_goalie['situation'] == 'all']
df_player = df_player[df_player['situation'] == 'all']

# TODO: TO IMPROVE POTENTIAL MODEL PERFORMACE, INCLUDE MULTIPLE SITUATIONS & MAYBE A HAVE A MODEL PROJECTION
#  WHEN THOSE SITUATIONS WILL HAPPEN IN A GIVEN GAME (PROJECTED PP, PK, OT, AND 4ON4 TIME PER GAME)

# New/adjusted stats
df_goalie['goalsSavedAboveX'] = df_goalie['xGoals'] - df_goalie['goals']
df_goalie['savePctg'] = (df_goalie['ongoal'] - df_goalie['goals']) / df_goalie['ongoal']
df_goalie['savePctgAboveX'] = (df_goalie['savePctg'] -
                               ((df_goalie['ongoal'] - df_goalie['xGoals']) / df_goalie['ongoal']))
df_goalie['gaa'] = (df_goalie['goals'] * 60) / (df_goalie['icetime'] / 60)
df_goalie['gaaAboveX'] = ((df_goalie['xGoals'] * 60) / (df_goalie['icetime'] / 60)) - df_goalie['gaa']
df_player['goalsAboveX'] = df_player['I_F_goals'] - df_player['I_F_xGoals']
df_player['shotPctg'] = df_player['I_F_goals'] / df_player['I_F_shotsOnGoal'] * 100
df_player['icetime/game'] = df_player['icetime'] / 60 / df_player['games_played']
i_f_columns = [col for col in df_player.columns if col.startswith('I_F_')]
# Per 60 stats
for col in i_f_columns:
    df_player[col + '_per_60'] = (df_player[col] / df_player['icetime'] * 60) * 60
df_player = df_player.drop(columns=i_f_columns)
df_player.columns = df_player.columns.str.replace('I_F_', '', regex=False).str.replace('_per_60', '/60')

# Commented out because already tested and does not need to be ran everytime
"""
# Plot graphs & print correlation
for col in df_player.columns:
    if col not in ['playerId', 'season', 'name', 'team', 'position', 'goals/60']:
        graph = sns.relplot(x=col, y='goals/60', data=df_player)
        graph.figure.subplots_adjust(top=0.9)
        graph.figure.suptitle(f'Goals vs {col}')
        plt.show()
        correlation = df_player['goals/60'].corr(df_player[col])
        print(f"Correlation between goals/60 and {col} is {correlation:.3f}")

Features can be above mentioned stats ^ as they look good and correlated 
Icetime/game, Icetime, and Games played are least important

# Outlier tests
# Z-scores
for col in df_player.columns:
    if col not in ['playerId', 'season', 'name', 'team', 'position', 'goals/60']:
        z_scores = zscore(df_player[col])
        potential_outliers = df_player[(z_scores > 3) | (z_scores < -3)]

        if not potential_outliers.empty:
            print(f"Outliers in {col}:")
            outlier_info = potential_outliers[[col]].copy()
            outlier_info['z_score'] = z_scores[np.abs(z_scores) > 3]
            outlier_info['index'] = potential_outliers.index
            outlier_info['name'] = potential_outliers['name']
            print(outlier_info)

There are a few outliers with a score of 4 or above, but generally there is not that many so will leave them in for now
LATER: Try the model without these potential outliers and see how it does (once with capping at 4, once capped at 3)
Honestly, all of it makes sense (the best goal scores tend to be leaders in here, or players who have a lot of high
danger chances and don't bury them 
"""

# Position as a dummy variable
df_player = pd.get_dummies(df_player, columns=['position'], prefix='is_a', dtype=int)
df_player.columns = df_player.columns.str.replace('is_a_', 'is_a_')
df_player.rename(columns={
    'is_a_C': 'is_a_C',
    'is_a_R': 'is_a_RW',
    'is_a_L': 'is_a_LW',
    'is_a_D': 'is_a_D'
}, inplace=True)

# Df_player is saved to sql, and can be called via the read_sql func from df_player
DATA_DIR = '/Users/chrisbugs/Downloads'
connector = sqlite3.connect(path.join(DATA_DIR, 'goal_model_data.sqlite'))
"""
df_player.to_sql('player_v1', connector, if_exists='replace', index=False)
"""
player_df = pd.read_sql(
    """
    SELECT * FROM player_v1
    """, connector)

# Define variables
"""
features_long = ['goalsAboveX', 'shotPctg', 'icetime/game', 'xGoals/60', 'xOnGoal/60', 'flurryAdjustedxGoals/60',
                 'shotsOnGoal/60', 'shotAttempts/60', 'savedUnblockedShotAttempts/60', 'highDangerShots/60',
                 'mediumDangerShots/60', 'lowDangerShots/60', 'highDangerxGoals/60', 'mediumDangerxGoals/60',
                 'lowDangerxGoals/60', 'highDangerGoals/60', 'mediumDangerGoals/60', 'lowDangerGoals/60',
                 'is_a_C', 'is_a_D', 'is_a_LW', 'is_a_RW', 'playerId']
features_short = ['goalsAboveX', 'shotPctg', 'icetime/game', 'flurryAdjustedxGoals/60',
                  'shotsOnGoal/60', 'highDangerShots/60', 'mediumDangerShots/60', 'lowDangerShots/60', 'playerId',
                  'highDangerxGoals/60', 'mediumDangerxGoals/60', 'lowDangerxGoals/60', 'highDangerGoals/60',
                  'mediumDangerGoals/60', 'lowDangerGoals/60', 'is_a_C', 'is_a_D', 'is_a_LW', 'is_a_RW']
"""
features = [
    'icetime/game', 'xGoals/60', 'xOnGoal/60', 'flurryAdjustedxGoals/60',
    'shotsOnGoal/60', 'shotAttempts/60', 'savedUnblockedShotAttempts/60',
    'highDangerShots/60', 'mediumDangerShots/60', 'lowDangerShots/60',
    'highDangerxGoals/60', 'mediumDangerxGoals/60', 'lowDangerxGoals/60',
    'is_a_C', 'is_a_D', 'is_a_LW', 'is_a_RW', 'playerId'
]

# TODO: NEED TO CHANGE THIS TARGET TO SOMETHING BETTER (MAYBE DO GOALS IN A SEASON OR SOMETHING) AND NEED TO
# TODO: MAKE SURE WHEN CHECKING MODEL SCORE TO HAVE THERE BE A WAY FOR THE MODEL TO CHECK ITS ACCURACY

# TODO: AND DO THE MONTE CARLO FOR FUN & PRACTICE!!!

target = 'goals/60'

X_train, X_test, y_train, y_test = train_test_split(player_df[features], player_df[target],
                                                    test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=4000,
                         learning_rate=0.04,
                         max_depth=3,
                         reg_lambda=2.5,
                         gamma=0,
                         eval_metric='logloss',
                         early_stopping_rounds=75)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
predictions = model.predict(X_test)

# Best params are below
"""
param_grid = {
    'max_depth': [3],
    'learning_rate': [0.04],
    'gamma': [0],
    'n_estimators': [4000],
    'reg_lambda': [2.5],
}
model = xgb.XGBRegressor()
# Setup GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
# Fit the model
grid_search.fit(X_train, y_train)
# Best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)
# Best estimator
best_model = grid_search.best_estimator_
# Make predictions with the best model
predictions = best_model.predict(X_test)
"""

feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print(feature_importances)
results = pd.DataFrame({'playerId': X_test['playerId'], 'Actual': y_test, 'Predicted': predictions})
df_w_pred = pd.merge(df_player.copy(), results, on='playerId')
df_w_pred['adjusted_pred'] = (df_w_pred['Predicted'] * df_w_pred['icetime/game'] / 60)
df_w_pred['adjusted_goals/60'] = (df_w_pred['goals/60'] * df_w_pred['icetime/game'] / 60)
df_w_pred['Season_total'] = df_w_pred['adjusted_goals/60'] * df_w_pred['games_played']
df_w_pred['Season_total_pred'] = df_w_pred['adjusted_pred'] * df_w_pred['games_played']
df_w_pred['Close'] = np.where(np.abs(df_w_pred['Season_total_pred'] - df_w_pred['Season_total']) < 1.4999, True, False)
# Fix this line
# df_w_pred['Close over 10'] = np.where(np.abs(df_w_pred['Season_total_pred'] - df_w_pred['Season_total']) < 1.4999 and
#                                      df_w_pred['Season_total'] > 10, True, False)


"""
# TTS for both feature styles
X_trainS, X_testS, y_trainS, y_testS = train_test_split(player_df[features_short],
                                                    player_df[target], test_size=0.2, random_state=42)

X_trainL, X_testL, y_trainL, y_testL = train_test_split(player_df[features_long],
                                                        player_df[target], test_size=0.2, random_state=42)

# Make and fit the models
modelS = xgb.XGBRegressor(n_estimators=250, learning_rate=0.1, max_depth=4)
modelS.fit(X_trainS, y_trainS)

modelL = xgb.XGBRegressor(n_estimators=250, learning_rate=0.1, max_depth=4)
modelL.fit(X_trainL, y_trainL)

# CV test the models
scoresS = cross_val_score(modelS, X_trainS, y_trainS, cv=5, scoring='neg_mean_squared_error')
#print("Cross-validation scores (MSE) of model S:", -scoresS)
#print("Mean MSE:", -scoresS.mean())
scoresL = cross_val_score(modelL, X_trainL, y_trainL, cv=5, scoring='neg_mean_squared_error')
#print("Cross-validation scores (MSE) of model L:", -scoresL)
#print("Mean MSE:", -scoresL.mean())

# Make predictions
y_predS = modelS.predict(X_testS)
y_predL = modelL.predict(X_testL)


# Check predictions
resultsS = pd.DataFrame({'playerId': X_testS['playerId'], 'Actual': y_testS, 'Predicted': y_predS})
resultsL = pd.DataFrame({'playerId': X_testL['playerId'], 'Actual': y_testL, 'Predicted': y_predL})
# Merge to adjust for avg icetime/game
S_df_with_pred = pd.merge(player_df.copy(), resultsS, on='playerId')
L_df_with_pred = pd.merge(player_df.copy(), resultsL, on='playerId')
S_df_with_pred['adjusted_pred'] = (S_df_with_pred['Predicted'] * S_df_with_pred['icetime/game'] / 60)
S_df_with_pred['adjusted_goals/60'] = (S_df_with_pred['goals/60'] * S_df_with_pred['icetime/game'] / 60)
S_df_with_pred['Season_total'] = S_df_with_pred['adjusted_goals/60'] * S_df_with_pred['games_played']
S_df_with_pred['Season_total_pred'] = S_df_with_pred['adjusted_pred'] * S_df_with_pred['games_played']
print(S_df_with_pred[['name', 'Season_total', 'Season_total_pred']].sample(20))
L_df_with_pred['adjusted_pred'] = (L_df_with_pred['Predicted'] * L_df_with_pred['icetime/game'] / 60)
L_df_with_pred['adjusted_goals/60'] = (L_df_with_pred['goals/60'] * L_df_with_pred['icetime/game'] / 60)
L_df_with_pred['Season_total'] = L_df_with_pred['adjusted_goals/60'] * L_df_with_pred['games_played']
L_df_with_pred['Season_total_pred'] = L_df_with_pred['adjusted_pred'] * L_df_with_pred['games_played']
# print(L_df_with_pred[['name', 'Season_total', 'Season_total_pred']].sample(20))


accuracyS = accuracy_score(y_testS, y_predS)
accuracyL = accuracy_score(y_testL, y_predL)
print(f"Model S accuracy score: {accuracyS}")
print(f"Model L accuracy score: {accuracyL}")

ll_score_S = log_loss(y_testS, y_predS)
ll_score_L = log_loss(y_testL, y_predL)
print(f"Model S log-loss score: {ll_score_S}")
print(f"Model L log-loss score: {ll_score_L}")


# Check feature importance
features_modelS = modelS.feature_importances_
features_modelL = modelL.feature_importances_
featuresS = features_short
featuresL = features_long
importance_S = pd.DataFrame({'feature': featuresS, 'importance': features_modelS})
importance_S = importance_S.sort_values(by='importance', ascending=False)
importance_L = pd.DataFrame({'feature': featuresL, 'importance': features_modelL})
importance_L = importance_L.sort_values(by='importance', ascending=False)
#print(f"Importances of model S\n{importance_S}")
#print(f"Importances of model L\n{importance_L}")
"""
