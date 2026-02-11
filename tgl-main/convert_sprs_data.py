import pandas as pd
import numpy as np
import torch
import argparse

import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
# parser.add_argument('--nss', type=str, help='negative sampling strategy name')
parser.add_argument('--ss', type=str, help='sparsification strategy name')
parser.add_argument('--upto', type=str, help='sparsification upto')
args=parser.parse_args()

load_location = '/raid/t2/TGN_adv/sparsified_data'
save_location = f'/raid/t2/TGN_adv/poisoned_data/{args.data}/{args.ss}'

# wikipedia_random_sparsified_0.7.csv
if args.ss == 'untouched':
    filename = f'{load_location}/{args.data}/{args.ss}/edges.csv'
else:
    filename = f'{load_location}/{args.data}/{args.ss}/{args.data}_{args.ss}_sparsified_{args.upto}.csv'

# filename = f'{load_location}/nss_{args.nss}/{args.data}/{args.ss}/{args.data}_{args.ss}_sparsified_{args.upto}.csv'

if os.path.exists(filename):
    print("The file exists", filename)
else:
    raise FileNotFoundError(f"The {filename} does not exist")


# Load the .csv file
df = pd.read_csv(filename)
df.rename(columns={'src':'u', 'dst': 'i', 'time': 'ts'}, inplace=True)

# data = np.load(f'{load_location}/nss_{args.nss}/{args.data}/{args.ss}/ml_{args.data}.npy')

# data = np.load(f'{load_location}/{args.data}/{args.ss}/ml_{args.data}.npy')

# print(data.shape)
# print(len(df['idx']))

# data = data[df['idx']]
# print(data.shape)

# tensor_data = torch.tensor(data)

# # TODO: edge features should be saved with ss as well upto.
# torch.save(tensor_data, f'{save_location}/edge_features.pt')
# torch.save(tensor_data, f'{save_location}/edge_features_{args.ss}_{args.upto}.pt')

# Convert the NumPy array to a PyTorch tensor
# tensor_data = torch.tensor(data)

# Drop the unnecessary columns
# print(df.columns)

# Always drop 'label' and 'idx'
df.drop(columns=['label', 'idx'], inplace=True)

# Conditionally drop any 'Unnamed' columns
unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
if unnamed_cols:
    df.drop(columns=unnamed_cols, inplace=True)
    print(f"Dropped unnamed columns: {unnamed_cols}")

# try:
#     df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'label', 'idx'], axis=1, inplace=True)
# except KeyError:
#     df.drop(['Unnamed: 0', 'label', 'idx'], axis=1, inplace=True)
# except Exception as e:
#     raise Exception(e, "Search Here")

# Calculate the indices for the splits
total_rows = len(df)
first_split = int(total_rows * 0.7)
second_split = int(total_rows * 0.85)

# tensor_data = torch.tensor(data)
# split_index = int(len(tensor_data)*0.7)
# first_half = tensor_data[:split_index]
# second_half = tensor_data[split_index:]
# first_half = first_half[:first_split]
# tensor_data = torch.cat((first_half, second_half), 0)   
# TODO: edge features should be saved with ss as well upto.
# tensor_data = np.round(tensor_data, 4)
# tensor_data = tensor_data[1:]
# torch.save(tensor_data, f'{save_location}/edge_features.pt')

# Create the 'ext_roll' column
df['ext_roll'] = 0
df.loc[first_split:second_split, 'ext_roll'] = 1
df.loc[second_split:, 'ext_roll'] = 2

if args.data.lower() == 'uci':
    print('UCI dataset, adding epoch = 1082040961')
    df['ts'] = df['ts'] + 1082040961
elif args.data.lower() == 'bitcoin':
    print('Bitcoin dataset, adding epoch = 1289241941')
    df['ts'] = df['ts'] + 1289241941



# Print all the columns
# print(df.columns)

# Optionally, print the first few rows to verify
# print(df.head())

df.rename(columns={'u': 'src', 'i': 'dst', 'ts': 'time'}, inplace=True)

if args.ss == 'untouched':
    df['time'] = df['time'].astype(int)
    df.to_csv(f'{save_location}/modified_{args.data}_{args.ss}_sparsified_{1.0}.csv', index=True)
else:
    df.to_csv(f'{save_location}/modified_sprs_only_{args.data}_{args.ss}_sparsified_{args.upto}.csv', index=True)

print('saved successfully at ', save_location)
