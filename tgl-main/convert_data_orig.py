import pandas as pd
import numpy as np
import torch

# Load the .csv file
df = pd.read_csv('./wiki_tspear2/edges_orig.csv')

# data = np.load('wiki_tspear2/edge.npy')

# print(data.shape)
# print(len(df['idx']))

# # data = data[df['idx']]
# # print(data.shape)
# tensor_data = torch.tensor(data)
# torch.save(tensor_data, './wiki_poison/edge_features.pt')

# # Convert the NumPy array to a PyTorch tensor
# tensor_data = torch.tensor(data)

# Drop the unnecessary columns
# df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'label', 'idx','is_negative'], axis=1, inplace=True)
df.drop(['Unnamed: 0', 'adv'], axis=1, inplace=True)

# Calculate the indices for the splits
total_rows = len(df)
first_split = int(total_rows * 0.7)
second_split = int(total_rows * 0.85)

# Create the 'ext_roll' column
df['ext_roll'] = 0
df.loc[first_split:second_split, 'ext_roll'] = 1
df.loc[second_split:, 'ext_roll'] = 2

# Print all the columns
print(df.columns)

# Optionally, print the first few rows to verify
print(df.head())

df.rename(columns={'u': 'src', 'i': 'dst', 'ts': 'time'}, inplace=True)

df.to_csv('./wiki_tspear2/edges.csv', index=True)
