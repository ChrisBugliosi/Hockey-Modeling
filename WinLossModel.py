import pandas as pd
import xgboost as xgb
import numpy as np
import sqlite3
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from os import path

# Load data
DATA_DIR = '/Users/chrisbugs/Downloads'
connector = sqlite3.connect(path.join(DATA_DIR, 'shift_team_rollingv2.sqlite'))

# Load all chunked tables and concatenate them into a single DataFrame
chunk_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' "
                           "AND name LIKE 'shift_team_dfv2_part%'",
                           connector)
chunk_dfs = [pd.read_sql(f"SELECT * FROM {table}", connector) for table in chunk_tables['name']]
df = pd.concat(chunk_dfs, axis=1)

df.reset_index(inplace=True)

# Calculate home_team_win only for 'all' situations
df['home_team_win'] = np.where(df['situation'] == 'all', df['goalsFor'] > df['away_goalsFor'], np.nan)

# Propagate the home_team_win result to all situations for the same game
df['home_team_win'] = df.groupby(['gameId'])['home_team_win'].transform(lambda x: x.ffill().bfill())

# Filter columns by keywords
def filter_columns_by_keywords(df, keywords):
    filtered_columns = [col for col in df.columns if any(keyword in col for keyword in keywords)]
    return filtered_columns

# Use 'sma' keyword
keywords = ['sma']

# Prepare features
df['situation'] = df['situation'].astype('category')
df = pd.get_dummies(df, columns=['situation'], drop_first=True)
dummy_columns = [col for col in df.columns if col.startswith('situation_')]

features = ['is_early_szn', 'is_mid_szn', 'is_late_szn'] + filter_columns_by_keywords(df, keywords) + dummy_columns
target = 'home_team_win'

# Sort by date to prevent leakage
df.sort_values(by='gameDate', inplace=True)

# Define holdout season (last season in this case)
holdout_season = df['season'].max()
train_test_df = df[df['season'] != holdout_season]
holdout_df = df[df['season'] == holdout_season]

X = train_test_df[features]
y = train_test_df[target]

X_holdout = holdout_df[features]
y_holdout = holdout_df[target]

# Define TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Comment out the Bayesian Optimization process

# # Bayesian Optimization function
# def xgb_evaluate(max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
#     params = {
#         'max_depth': int(max_depth),
#         'learning_rate': learning_rate,
#         'n_estimators': int(n_estimators),
#         'gamma': gamma,
#         'min_child_weight': min_child_weight,
#         'subsample': subsample,
#         'colsample_bytree': colsample_bytree,
#         'eval_metric': 'logloss'
#     }

#     tscv = TimeSeriesSplit(n_splits=5)
#     accuracies = []
#     log_losses = []

#     for train_index, test_index in tscv.split(X):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         xgb_classifier = xgb.XGBClassifier(**params)
#         xgb_classifier.fit(X_train, y_train)

#         y_pred_proba_class = xgb_classifier.predict_proba(X_test)[:, 1]

#         logloss_class = log_loss(y_test, y_pred_proba_class)
#         log_losses.append(logloss_class)

#     return -np.mean(log_losses)  # Negative because Bayesian Optimization maximizes

# # Define parameter ranges
# params = {
#     'max_depth': (3, 10),
#     'learning_rate': (0.01, 0.3),
#     'n_estimators': (50, 300),
#     'gamma': (0, 1),
#     'min_child_weight': (1, 10),
#     'subsample': (0.6, 1),
#     'colsample_bytree': (0.6, 1)
# }

# # Run Bayesian Optimization
# optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=params, random_state=0)
# optimizer.maximize(init_points=10, n_iter=50)

# # Get best parameters
# best_params = optimizer.max['params']
# best_params['max_depth'] = int(best_params['max_depth'])
# best_params['n_estimators'] = int(best_params['n_estimators'])

# Replace with your best parameters from the optimization
best_params = {
    'colsample_bytree': 1.0,
    'gamma': 1.0,
    'learning_rate': 0.01,
    'max_depth': 10,
    'min_child_weight': 10.0,
    'n_estimators': 156,
    'subsample': 0.6,
    'eval_metric': 'logloss'
}

# Evaluate with best parameters
xgb_classifier = xgb.XGBClassifier(**best_params)
xgb_classifier.fit(X, y)

y_holdout_pred_class = xgb_classifier.predict(X_holdout)
y_holdout_pred_proba_class = xgb_classifier.predict_proba(X_holdout)[:, 1]

holdout_accuracy = accuracy_score(y_holdout, y_holdout_pred_class)
holdout_logloss = log_loss(y_holdout, y_holdout_pred_proba_class)
holdout_confusion_matrix = confusion_matrix(y_holdout, y_holdout_pred_class)

print(f"Best Parameters: {best_params}")
print(f"Holdout Season Accuracy: {holdout_accuracy}")
print(f"Holdout Season Log Loss: {holdout_logloss}")
print(f"Confusion Matrix for Holdout Season:\n{holdout_confusion_matrix}")



# Feature importances
feature_importances = pd.DataFrame({'feature': features, 'importance': xgb_classifier.feature_importances_})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
print(feature_importances.head(50))


"""
Best Parameters: {'colsample_bytree': 1.0, 'gamma': 1.0, 'learning_rate': 0.01, 'max_depth': 10, 
    'min_child_weight': 10.0, 'n_estimators': 156, 'subsample': 0.6}
Holdout Season Accuracy: 0.5505714285714286
Holdout Season Log Loss: 0.6858783967489863
Confusion Matrix for Holdout Season:
[[2328 1152]
 [1994 1526]]
"""