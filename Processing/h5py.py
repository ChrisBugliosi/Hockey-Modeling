import sqlite3
import h5py
import numpy as np
import pandas as pd
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

# Step 1: Convert DataFrame to a NumPy array
data_array = df.to_numpy()

# Step 2: Save the data to an HDF5 file using h5py
with h5py.File(path.join(DATA_DIR, 'final_team_data.h5'), 'w') as hdf5_file:
    # Step 3: Store the entire DataFrame as a dataset in HDF5
    hdf5_file.create_dataset('final_team_data', data=data_array)

# Close the SQLite connection
connector.close()
