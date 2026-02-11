import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import  pickle
from pathlib import Path
import torch

scratch_location = '/raid/t2/TGN_adv'


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="WIKI")
args=parser.parse_args()


save_location = Path(f'{scratch_location}/tspear/DATA/{args.data.upper()}')

save_location.mkdir(parents=True, exist_ok=True)

if args.data.lower() == 'wiki':
    load_location = f'{scratch_location}/DyGlib/DG_data/wikipedia'
    args.data = 'wikipedia'
elif args.data.lower() == 'socialevo':
    load_location = f'{scratch_location}/DyGlib/DG_data/SocialEvo'
    args.data = 'SocialEvo'
else:
    load_location = f'{scratch_location}/DyGlib/DG_data/{args.data.lower()}'
    args.data = args.data.lower()

df = pd.read_csv(f'{load_location}/ml_{args.data}.csv')

if args.data.lower() == 'bitcoin':
    df = pd.read_csv(f'{load_location}/ml_{args.data}.csv',
                    names=['u', 'i', 'ts', 'label', 'idx'], 
                 header=0, skiprows=1)
else:
    df = pd.read_csv(f'{load_location}/ml_{args.data}.csv')


df.rename(columns={'u': 'src', 'i': 'dst', 'ts': 'time'}, inplace=True)

if all(col in df.columns for col in ['Unnamed: 0', 'label', 'idx']):
    df.drop(['Unnamed: 0', 'label', 'idx'], axis=1, inplace=True)


# Calculate the indices for the splits
total_rows = len(df)
first_split = int(total_rows * 0.7)
second_split = int(total_rows * 0.85)

# Create the 'ext_roll' column
df['ext_roll'] = 0
df.loc[first_split:second_split, 'ext_roll'] = 1
df.loc[second_split:, 'ext_roll'] = 2

if args.data.lower() == 'uci':
    df['time'] = df['time'] + 1082040961
    

df.to_csv(f'{save_location}/edges.csv', index=True)


#  to save npz file

num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

ext_full_indptr = np.zeros(num_nodes + 1, dtype=int)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['src'])
    dst = int(row['dst'])
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['time'])
    ext_full_eid[src].append(idx)
    ext_full_indices[dst].append(src)
    ext_full_ts[dst].append(row['time'])
    ext_full_eid[dst].append(idx)

def ext_sort(i, indices, t, eid):
    idx = np.argsort(t[i])
    indices[i] = np.array(indices[i])[idx].tolist()
    t[i] = np.array(t[i])[idx].tolist()
    eid[i] = np.array(eid[i])[idx].tolist()

for i in tqdm(range(len(ext_full_ts)), disable=False):
    ext_sort(i, ext_full_indices, ext_full_ts, ext_full_eid)

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

np_ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
np_ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
np_ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx]

for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
    tsort(i, ext_full_indptr, np_ext_full_indices, np_ext_full_ts, np_ext_full_eid)

g = {
    'indptr': np.array(ext_full_indptr),  # made it np.array
    'indices': np_ext_full_indices,
    'ts': np_ext_full_ts,
    'eid': np_ext_full_eid,
    'ext_full_indices': ext_full_indices,
    'ext_full_ts': ext_full_ts,
    'ext_full_eid': ext_full_eid
}

with open(f'{save_location}/ext_full.pkl', 'wb') as f:
    pickle.dump(g, f)
    print(f'dumped at {save_location}/ext_full.pkl')

if args.data.lower() == 'wikipedia':
    # this happens only for wikipedia right now.
    data = np.load(f'{load_location}/ml_{args.data}.npy')
    tensor_data = torch.tensor(data)
    tensor_data = np.round(tensor_data, 4)
    tensor_data = tensor_data[1:]
    torch.save(tensor_data, f'{save_location}/edge_features.pt')
    print('saved edge_features.pt')
print('done')
