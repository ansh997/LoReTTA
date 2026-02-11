#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home2/hmnshpl/projects/results/tspear/downloading_GDELT.txt
#SBATCH --nodelist gnode090
#SBATCH --time=96:00:00
#SBATCH --mail-user=himanshu.pal@research.iiit.ac.in
#SBATCH --mail-type=ALL

wget -P /raid/t2/TGN_adv/tspear/DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/int_train.npz           
wget -P /raid/t2/TGN_adv/tspear/DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/int_full.npz             
wget -P /raid/t2/TGN_adv/tspear/DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt     
wget -P /raid/t2/TGN_adv/tspear/DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv                
wget -P /raid/t2/TGN_adv/tspear/DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/ext_full.npz            
wget -P /raid/t2/TGN_adv/tspear/DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv                  
wget -P /raid/t2/TGN_adv/tspear/DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edge_features.pt
