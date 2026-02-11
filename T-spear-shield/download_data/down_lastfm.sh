#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home2/hmnshpl/projects/results/tspear/downloading_lastfm.txt
#SBATCH --nodelist gnode090
#SBATCH --time=96:00:00
#SBATCH --mail-user=himanshu.pal@research.iiit.ac.in
#SBATCH --mail-type=ALL

wget -P /raid/t2/TGN_adv/tspear/DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/edges.csv                 
wget -P /raid/t2/TGN_adv/tspear/DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/ext_full.npz           
wget -P /raid/t2/TGN_adv/tspear/DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/int_full.npz          
wget -P /raid/t2/TGN_adv/tspear/DATA/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/int_train.npz        
