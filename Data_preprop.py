import pandas as pd
import numpy as np
import requests
import sqlite3
from os import path
import dask.dataframe as dd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

# TODO: CALCULATE ROLLING STATISTICS AND DROP USELESS COLUMNS & TRAIN A MODEL USING JUST ALL_TEAMS CSV
#   AND ON A SIDE NOTE WORK ON GETTING OPPOSING TEAM DATA FROM GAME LOGS FROM THE NHL API, OR ELSE USE BASIC
#   OPPOSING TEAM STATISTICS (ONES THAT ARE AVAILABLE FROM THE CSV - LINE UP GAME IDS)
#   FOR THE SIDE NOTE: TRY TO GET BOXSCORE API CALL FOR OPPOSING TEAM STATS, MIGHT NEED TO USE IT TO GET
#   FINAL SCORE OF THE GAME

# TODO: MAKE MEGA DF AND MAKE A FUNC THAT MATCHES GAMEID/GAMEDATE FROM https://api-web.nhle.com/v1/score/2008-10-04
#  TO EXISITING DF AND THEN FLATTEN AND DROP ALL THE EXCESS STUFF - ASSING SCORE BY ABRV TO AVOID DUPLICATE?

# TODO: IDEA: CREATE A COPY DF WITH ALL GAME IDS AND GAME DATES OF THE HOME TEAM, HAVE THE API CALLS TO SCORES GET HOME
#  TEAMS SCORE OF GAME, THEN HAVE THE MAIN DATAFRAME BE ONLY HOME TEAMS AND THEN HAVE A SECOND DF BE ALL THE AWAY TEAMS
#  AND THEN MERGE THE DFS ON GAME ID, AND HAVE ALL THE AWAY TEAM DF'S COLUMNS BE CALLED OPP_(INSERT STAT HERE) SO THAT
#  YOU HAVE HOME TEAM AND OPPOSING TEAM STATS, AND THEN HAVE THE MODEL PREDICT WHETHER OR NOT THE HOME TEAM WILL WIN

# TODO: DONT FORGET ABOUT DATA CLEANING (OUTLIER REMOVAL) - is this needed?

# TODO: NO NEED FOR API SCORES CALL, DATAFRAME ALREADY HAS GOALS FOR AND GOALS AGAINST INCLUDED, HOWEVER THE SPLITTING
#  BY HOME AND AWAY IS STILL A GOOD IDEA (MERGE ON GAMEID AND SITUATION, MAKE AWAY TEAM STATS = OPP_(STAT), TRY TO GET
#  STANDINGS INFO IF POSSIBLE - NEED A FUNC TO DO THIS AND GET HOME AND AWAY TEAM STANDING)

# flattens game data
def flatten_game_data(nested):
    flat = {key: value for key, value in nested.items() if type(value) is not dict}
    if 'commonName' in nested.keys():
        flat['teamName'] = nested['commonName']['default']
    if 'opponentCommonName' in nested.keys():
        flat['oppTeamName'] = nested['opponentCommonName']['default']
    if 'gameId' in nested.keys():
        flat['gameId'] = str(nested['gameId'])
    return flat

# flattens player data from api
def flatten_player(nested):
    flat = {key: value for key, value in nested.items() if type(value) not in (dict, list)}
    flat['name'] = nested['firstName']['default'] + nested['lastName']['default']
    return flat

# gets player game-by-game data for a given id
def get_game_data(player_id, season):
    stats_url = f'https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2'
    stats_resp = requests.get(stats_url)
    stats_json = stats_resp.json()
    return pd.DataFrame([flatten_game_data(x) for x in stats_json['gameLog']])

# returns the player game-by-game data for a team
def team_player_data(team_df, season):
    all_player_data = []
    for player_id in team_df['id']:
        game_data = get_game_data(player_id, season)
        game_data['playerId'] = player_id
        all_player_data.append(game_data)
    full_team_df = pd.concat(all_player_data, ignore_index=True)
    return full_team_df

# returns player's game-by-game data for a team and season (up to most recent date)
def player_game_by_game(team, season):
    roster_url = f'https://api-web.nhle.com/v1/roster/{team}/{season}'
    roster_resp = requests.get(roster_url)
    roster_json = roster_resp.json()

    df_fwd = pd.DataFrame([flatten_player(x) for x in roster_json['forwards']])
    df_def = pd.DataFrame([flatten_player(x) for x in roster_json['defensemen']])
    df_g = pd.DataFrame([flatten_player(x) for x in roster_json['goalies']])

    df_players = pd.concat([df_fwd, df_def], ignore_index=True)
    df_players = team_player_data(df_players, season)
    df_g = team_player_data(df_g, season)

    df_all = pd.concat([df_players, df_g], ignore_index=True)
    df_all['team'] = team

    return df_all

# sample 23-24 toronto team stats
# tor_df = player_game_by_game('TOR', 20232024)
"""
# Optimize Data Types
def optimize_dtypes(df):
    int_cols = df.select_dtypes(include=['int64']).columns
    float_cols = df.select_dtypes(include=['float64']).columns
    df[int_cols] = df[int_cols].astype('int32')
    df[float_cols] = df[float_cols].astype('float32')
    return df

# Weighted Moving Average Function
def weighted_moving_average(x, window):
    weights = np.arange(1, window + 1)
    return np.dot(x, weights) / weights.sum() if len(x) == window else np.nan

# Function to calculate SMA, WMA, and EMA by situation using Dask
def calculate_moving_averages(df, columns, ema_spans, rolling_windows):
    new_columns_list = []
    for column in columns:
        for situation in df['situation'].unique():
            situation_df = df[df['situation'] == situation]
            for window in rolling_windows:
                new_columns_list.append(situation_df[column].shift(1).rolling(
                window=window, min_periods=1).mean().rename(f'sma_{column}_rolling_{window}_{situation}'))
                new_columns_list.append(situation_df[column].shift(1).rolling(window=window).apply(lambda x: 
                weighted_moving_average(x, window), raw=False).rename(f'wma_{column}_rolling_{window}_{situation}'))
            new_columns_list.append(situation_df[column].shift(1).ewm(
            span=ema_spans['early'], adjust=False).mean().rename(f'ema_early_{column}_{situation}'))
            new_columns_list.append(situation_df[column].shift(1).ewm(
            span=ema_spans['mid'], adjust=False).mean().rename(f'ema_mid_{column}_{situation}'))
            new_columns_list.append(situation_df[column].shift(1).ewm(
            span=ema_spans['late'], adjust=False).mean().rename(f'ema_late_{column}_{situation}'))
    new_columns_df = pd.concat(new_columns_list, axis=1)
    result_df = pd.concat([df, new_columns_df], axis=1)
    return result_df

# Define EMA spans for each season phase
ema_spans = {
    'early': 3,
    'mid': 5,
    'late': 8
}

# Define rolling windows
rolling_windows = [5, 10, 15, 25, 40]

# Load your DataFrame with Dask
df_teamlevel_raw = dd.read_csv(path.join('/Users/chrisbugs/Downloads/all_teams.csv'))

# Convert Dask DataFrame to Pandas DataFrame for processing
df = df_teamlevel_raw.compute()

# Optimize data types
df = optimize_dtypes(df)

# create the df with desired columns
df = df[['team', 'season', 'gameId', 'home_or_away', 'gameDate', 'situation',
         'iceTime', 'xOnGoalFor', 'xGoalsFor', 'flurryAdjustedxGoalsFor', 'shotsOnGoalFor',
         'shotAttemptsFor', 'goalsFor', 'savedShotsOnGoalFor', 'penalityMinutesFor',
         'faceOffsWonFor', 'hitsFor', 'takeawaysFor', 'giveawaysFor', 'lowDangerShotsFor',
         'mediumDangerShotsFor', 'highDangerShotsFor', 'lowDangerxGoalsFor', 'mediumDangerxGoalsFor',
         'highDangerxGoalsFor', 'lowDangerGoalsFor', 'mediumDangerGoalsFor', 'highDangerGoalsFor',
         'dZoneGiveawaysFor', 'playoffGame']]

# Sort the DataFrame by team, season, and gameDate
df = df.sort_values(by=['team', 'season', 'gameDate'])
# Group by team and season, then create a cumulative count of games
df['gameNumber'] = df.groupby(['team', 'season', 'situation']).cumcount() + 1
# Early/mid/late szn stats, feel free to alter this later on and try with/without it (CONSIDER BASING OFF EMA SPAN)
df['is_early_szn'] = df['gameNumber'].apply(lambda x: 1 if x <= 25 else 0)
df['is_mid_szn'] = df['gameNumber'].apply(lambda x: 1 if 26 <= x <= 51 else 0)
df['is_late_szn'] = df['gameNumber'].apply(lambda x: 1 if x >= 52 else 0)

# split big df into a bunch of smaller dfs inside a dictionary
team_dfs = {team: df[df['team'] == team].copy() for team in df['team'].unique()}

# Define columns to calculate
columns_to_calculate = ['iceTime', 'xOnGoalFor', 'xGoalsFor', 'flurryAdjustedxGoalsFor', 'shotsOnGoalFor',
                        'shotAttemptsFor', 'goalsFor', 'savedShotsOnGoalFor', 'penalityMinutesFor',
                        'faceOffsWonFor', 'hitsFor', 'takeawaysFor', 'giveawaysFor', 'lowDangerShotsFor',
                        'mediumDangerShotsFor', 'highDangerShotsFor', 'lowDangerxGoalsFor', 'mediumDangerxGoalsFor',
                        'highDangerxGoalsFor', 'lowDangerGoalsFor', 'mediumDangerGoalsFor', 'highDangerGoalsFor',
                        'dZoneGiveawaysFor']

# Run the function in smaller chunks
result_df = pd.DataFrame()
for team, df_chunk in team_dfs.items():
    chunk_result = calculate_moving_averages(df_chunk, columns_to_calculate, ema_spans, rolling_windows)
    result_df = pd.concat([result_df, chunk_result])

# Save the result
result_df.to_csv('/Users/chrisbugs/Downloads/shift_rollingv2.csv')
"""
def calculate_correlation(df, columns, target_column='goalsFor', num_values=100):
    correlation_results = {}

    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Calculate correlation for each column in the list
    for column in columns:
        if column in df.columns:
            correlation = df[column].iloc[:num_values].corr(df[target_column].iloc[:num_values])
            correlation_results[column] = correlation
        else:
            correlation_results[column] = None
            print(f"Column '{column}' not found in DataFrame.")

    return correlation_results

columns_to_calculate = ['sma_xGoalsFor_rolling_5_all', 'wma_xGoalsFor_rolling_5_all', 'sma_xGoalsFor_rolling_10_all',
                        'wma_xGoalsFor_rolling_10_all', 'ema_early_xGoalsFor_all', 'ema_mid_xGoalsFor_all',
                        'ema_late_xGoalsFor_all']

df = pd.read_csv('/Users/chrisbugs/Downloads/shift_rollingv2.csv')

# correlations = calculate_correlation(df, columns_to_calculate)
# print(correlations)

# create home and away dfs
home_df = (df[df['home_or_away'] == 'HOME']).reset_index(drop=True)
away_df = (df[df['home_or_away'] == 'AWAY']).reset_index(drop=True)

# rename away df columns
columns_excluded = ['season', 'gameId', 'gameDate', 'situation', 'home_or_away', 'playoffGame']

def add_away_prefix(df, exclude_columns):
    # Create a dictionary for renaming columns
    rename_dict = {col: f'away_{col}' for col in df.columns if col not in exclude_columns}

    # Rename the columns
    df_renamed = df.rename(columns=rename_dict)

    return df_renamed

away_df = add_away_prefix(away_df, columns_excluded)

# merge the dfs
df = pd.merge(home_df, away_df, on=['gameId', 'situation', 'playoffGame', 'season', 'gameDate'], how='inner')
df.drop(['home_or_away_x', 'home_or_away_y'], inplace=True, axis=1)

# Calculate home_team_win only for 'all' situations
df['home_team_win'] = np.where(df['situation'] == 'all', df['goalsFor'] > df['away_goalsFor'], np.nan)

# Propagate the home_team_win result to all situations for the same game
df['home_team_win'] = df.groupby(['gameId'])['home_team_win'].transform(lambda x: x.ffill().bfill())

df.to_csv('/Users/chrisbugs/Downloads/home_away_shift_dfv2.csv', index=False)


DATA_DIR = '/Users/chrisbugs/Downloads'
connector = sqlite3.connect(path.join(DATA_DIR, 'shift_team_rollingv2.sqlite'))
# df.to_sql('shift_team_dfv2', connector, if_exists='replace', index=False)

# Save the result in smaller chunks to avoid too many columns error
chunk_size = 1000
for i in range(0, len(df.columns), chunk_size):
    chunk_df = df.iloc[:, i:i + chunk_size]
    table_name = f'shift_team_dfv2_part{i // chunk_size}'
    chunk_df.to_sql(table_name, connector, if_exists='replace', index=False)

print(df.size)
