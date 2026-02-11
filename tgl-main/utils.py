import pickle
import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np

load_location = '/raid/t2/TGN_adv/poisoned_data'

def load_attacked_feat(d, rand_de=0, rand_dn=0, args=None):
    node_feats = None
    if os.path.exists(f'/raid/t2/TGN_adv/tspear/DATA/{d}/ATTACKED/{args.attack}_TGN_{args.ptb_rate}_True_0/node_features.pt'):
        node_feats = torch.load(f'/raid/t2/TGN_adv/tspear/DATA/{d}/ATTACKED/{args.attack}_TGN_{args.ptb_rate}_True_0/node_features.pt')
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists(f'/raid/t2/TGN_adv/tspear/DATA/{d}/ATTACKED/{args.attack}_TGN_{args.ptb_rate}_True_0/edge_features.pt'):
        edge_feats = torch.load(f'/raid/t2/TGN_adv/tspear/DATA/{d}/ATTACKED/{args.attack}_TGN_{args.ptb_rate}_True_0/edge_features.pt')
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(int(1293103*1.3), rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(int(411749*1.3), rand_de)
        elif d == "UCI":
            edge_feats = torch.randn(int(59835*1.3), rand_de)
        elif d == "BITCOIN":
            edge_feats = torch.randn(int(35590*1.3), rand_de)
        elif d == "ENRON":
            edge_feats = torch.randn(int(125236*1.3), rand_de)
        elif d == "SOCIALEVO":
            edge_feats = torch.randn(int(2099520*1.3), rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            node_feats = torch.randn(7144, rand_dn)
        elif d == "UCI":
            node_feats = torch.randn(1900, rand_dn)
        elif d == "BITCOIN":
            node_feats = torch.randn(6006, rand_dn)
        elif d == "ENRON":
            node_feats = torch.randn(int(185), rand_de)
        elif d == "SOCIALEVO":
            node_feats = torch.randn(int(75), rand_de)
    return node_feats, edge_feats

def load_attacked_graph(d, args):
    df = pd.read_csv(f'/raid/t2/TGN_adv/tspear/DATA/{d}/ATTACKED/{args.attack}_TGN_{args.ptb_rate}_True_0/edges.csv')
     
    with open(f'/raid/t2/TGN_adv/tspear/DATA/{d}/ATTACKED/{args.attack}_TGN_{args.ptb_rate}_True_0/ext_full.pkl', 'rb') as f:
        g = pickle.load(f)     
    return g, df

def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('/raid/t2/TGN_adv/tspear/DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('/raid/t2/TGN_adv/tspear/DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('/raid/t2/TGN_adv/tspear/DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('/raid/t2/TGN_adv/tspear/DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
        elif d == "UCI":
            edge_feats = torch.randn(59835, rand_de)
        elif d == "BITCOIN":
            edge_feats = torch.randn(35590, rand_de)
        elif d == "ENRON":
            edge_feats = torch.randn(int(125236), rand_de)
        elif d == "SOCIALEVO":
            edge_feats = torch.randn(int(2099520), rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            node_feats = torch.randn(7144, rand_dn)
        elif d == "UCI":
            node_feats = torch.randn(1900, rand_dn)
        elif d == "BITCOIN":
            node_feats = torch.randn(6006, rand_dn)
        elif d == "ENRON":
            node_feats = torch.randn(int(185), rand_de)
        elif d == "SOCIALEVO":
            node_feats = torch.randn(int(75), rand_de)
    return node_feats, edge_feats

def load_poisoned_feat(d, rand_de=0, rand_dn=0, ss='ts_tpr_remove_mss', upto=0.7):
    node_feats = None
    # edge_features_ts_tpr_remove_mss_0.7.pt
    
    # print('/raid/t2/TGN_adv/poisoned_data/{}/{}/edge_features_{}_{}.pt'.format(d, ss, ss, upto))
    print(os.path.exists('/raid/t2/TGN_adv/poisoned_data/{}/{}/edge_features.pt'.format(d, ss, ss, upto)))
    
    if os.path.exists('/raid/t2/TGN_adv/poisoned_data/{}/{}/node_features.pt'.format(d, ss, ss, upto)):
        node_feats = torch.load('/raid/t2/TGN_adv/poisoned_data/{}/{}/node_features.pt'.format(d, ss, ss, upto))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('/raid/t2/TGN_adv/poisoned_data/{}/{}/edge_features.pt'.format(d, ss, ss, upto)):
        edge_feats = torch.load('/raid/t2/TGN_adv/poisoned_data/{}/{}/edge_features.pt'.format(d, ss, ss, upto))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            node_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats

def old_load_graph(d):
    df = pd.read_csv('/raid/t2/TGN_adv/tspear/DATA/{}/edges.csv'.format(d))
    if d != 'wiki_tspear':
        g = np.load('/raid/t2/TGN_adv/tspear/DATA/{}/ext_full.npz'.format(d)) # special case for wiki_tspear
    else:
        # g = np.load('/raid/t2/TGN_adv/tspear/DATA/{}/ext_full.npz'.format(d)) # special case for wiki_tspear   
        print('loading for tspear', end='\r')
        with open(f'/raid/t2/TGN_adv/tspear/DATA/{d}/ext_full.pkl', 'rb') as f:
            g = pickle.load(f)     
        print('Loaded for tspear      ')
    return g, df

def load_graph(d):
    df = pd.read_csv('/raid/t2/TGN_adv/tspear/DATA/{}/edges.csv'.format(d))
     
    # with open(f'/raid/t2/TGN_adv/tspear/DATA/{d}/ext_full.pkl', 'rb') as f:
    #     g = pickle.load(f) 
      
    g = np.load('/raid/t2/TGN_adv/tspear/DATA/{}/ext_full.npz'.format(d), allow_pickle=True)
    return g, df

def load_poisoned_graph(d, nss, ss, upto, flag=False):
    if flag:
        save_location = f'{load_location}/nss_{nss}/{d}/{ss}'
    else:
        save_location = f'{load_location}/{d}/{ss}'
    df = pd.read_csv(f'{save_location}/modified_{d}_{ss}_sparsified_{upto}.csv')
    if d.lower() in ['reddit', 'wikipedia', 'mooc']:
        print('loading npz')
        g = np.load(f'{save_location}/modified_{d}_{ss}_sparsified_{upto}_ext_full.npz') # maybe for each poisoned dataset we produce a ext_full.npz
    else:
        print('loading pkl')
        with open(f'{save_location}/ext_full.pkl', 'rb') as f:
            g = pickle.load(f)
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs

def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].long()].float()
                        b.edata['f'] = srch.cuda()
    return mfgs

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].long())
    else:
        eids = None
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs

