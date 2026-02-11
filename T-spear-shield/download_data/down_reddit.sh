# !/bin/bash
# SBATCH -A research
# SBATCH -n 9
# SBATCH --gres=gpu:1
# SBATCH --mem-per-cpu=2G
# SBATCH --output=/home2/hmnshpl/projects/results/tspear/downloading_reddit.txt
# SBATCH --nodelist gnode090
# SBATCH --time=96:00:00
# SBATCH --mail-user=himanshu.pal@research.iiit.ac.in
# SBATCH --mail-type=ALL

wget -P /raid/t2/TGN_adv/tspear/DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features.pt
wget -P /raid/t2/TGN_adv/tspear/DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges.csv
wget -P /raid/t2/TGN_adv/tspear/DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/ext_full.npz
wget -P /raid/t2/TGN_adv/tspear/DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/int_full.npz
wget -P /raid/t2/TGN_adv/tspear/DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/int_train.npz
wget -P /raid/t2/TGN_adv/tspear/DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv
