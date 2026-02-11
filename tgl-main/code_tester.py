import pickle
import numpy as np
import argparse

data_dir = '/raid/t2/TGN_adv/tspear/DATA/'

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
args=parser.parse_args()

# Define paths for loading and saving
pkl_path = f'{data_dir}/{args.data}/ext_full.pkl'
npz_path = f'{data_dir}/{args.data}/ext_full.npz'

# Load the .pkl file
with open(pkl_path, 'rb') as f:
    g = pickle.load(f)

# Save the relevant arrays into a .npz file
np.savez(npz_path, indptr=np.array(g['indptr']), indices=np.array(g['ext_full_indices']), ts=np.array(g['ext_full_ts']), eid=np.array(g['ext_full_eid']))

print(f'Data saved to {npz_path}')
