import pandas as pd
import numpy as np
import torch
import argparse

import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--nss', type=str, help='negative sampling strategy name')
parser.add_argument('--ss', type=str, help='sparsification strategy name')
parser.add_argument('--upto', type=str, help='sparsification upto')
args=parser.parse_args()

load_location = '/raid/t2/TGN_adv/poisoned_data'
save_location = f'{load_location}/nss_{args.nss}/{args.data}/{args.ss}'

# wikipedia_random_sparsified_0.7.csv

filename = f'{load_location}/nss_{args.nss}/{args.data}/{args.ss}/{args.data}_{args.ss}_sparsified_{args.upto}.csv'

if os.path.exists(filename):
    print("The file exists", filename)
else:
    raise FileNotFoundError(f"The {filename} does not exist")

# assert 0 == 1, "intentional...."

# Load the .csv file
df = pd.read_csv(filename)

data = np.load(f'{load_location}/nss_{args.nss}/{args.data}/{args.ss}/ml_{args.data}.npy')

# print(data.shape)
# print(len(df['idx']))

# data = data[df['idx']]
# print(data.shape)
tensor_data = torch.tensor(data)
torch.save(tensor_data, f'{save_location}/edge_features.pt')

# Convert the NumPy array to a PyTorch tensor
tensor_data = torch.tensor(data)

# Drop the unnecessary columns
# print(df.columns)

try:
    df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'label', 'idx'], axis=1, inplace=True)
except KeyError:
    df.drop(['Unnamed: 0', 'label', 'idx'], axis=1, inplace=True)
except Exception as e:
    raise Exception(e, "Search Here")

# Calculate the indices for the splits
total_rows = len(df)
first_split = int(total_rows * 0.7)
second_split = int(total_rows * 0.85)

# Create the 'ext_roll' column
df['ext_roll'] = 0
df.loc[first_split:second_split, 'ext_roll'] = 1
df.loc[second_split:, 'ext_roll'] = 2

# Print all the columns
# print(df.columns)

# Optionally, print the first few rows to verify
# print(df.head())

df.rename(columns={'u': 'src', 'i': 'dst', 'ts': 'time'}, inplace=True)

df.to_csv(f'{save_location}/modified_{args.data}_{args.ss}_sparsified_{args.upto}.csv', index=True)

print('saved successfully at ', save_location)
