from pathlib import Path
import pickle
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import os
import time

from tatk.utils import (
    set_random_seed,
    parse_argument,
    parse_config,
    pre_exp_setting,
    load_data,
    get_attacked_data_dir,
    load_attacked_data
)
from tatk.attacker_C3 import TemporalAttack
from tatk.utils_train import (
    create_model_mailbox_sampler,
    train_model_link_pred,
    robust_train_model_link_pred_v2,
)

EXP_ID = 'AAAI'

def train_main(args):
    pre_exp_setting(EXP_ID, args)
    
    print(args)

    val_aps, val_aucs, val_hits = [], [], []
    test_aps, test_aucs, test_hits = [], [], []

    for i, seed in enumerate(args.seeds):
        set_random_seed(seed)

        if args.attack == "none":
            node_feats, edge_feats, g, df = load_data(args.data, args)
        else:
            node_feats, edge_feats, g, df = load_attacked_data(args.data, args)  
        
        ## Model train
        sample_param, memory_param, gnn_param, train_param = parse_config(args.model)
        model, mailbox, sampler = create_model_mailbox_sampler(node_feats, edge_feats, g, df, sample_param, memory_param, gnn_param, train_param)

        print(f'{args.robust = }')
        if args.robust == "none" or args.robust == "svd":
            val_ap, val_auc, val_hit, test_ap, test_auc, test_hit = train_model_link_pred(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, seed=seed)
        elif args.robust == "cosine":
            val_ap, val_auc, val_hit, test_ap, test_auc, test_hit,final_avg = robust_train_model_link_pred_v2(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, seed=seed)
        elif args.robust == "proposed":
            val_ap, val_auc, val_hit, test_ap, test_auc, test_hit,final_avg = robust_train_model_link_pred_v2(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, seed=seed)

        val_aps.append(val_ap)
        val_aucs.append(val_auc)
        val_hits.append(val_hit)
        test_aps.append(test_ap)
        test_aucs.append(test_auc)
        test_hits.append(test_hit)
        args.logger.info(f'[Seed {seed}] model: {args.model}, attack: {args.attack} ({args.ptb_rate:.2f}), val_auc: {val_auc:.4f}, val_hit: {val_hit:.4f}, test_ap: {test_ap:.4f}, test_auc: {test_auc:.4f}, test_hit: {test_hit:.4f}, neg_samples: {args.eval_neg_samples}')

        if node_feats is not None:
            del node_feats
        if edge_feats is not None:
            del edge_feats
        del g, df
        
    val_ap_mean, val_ap_std = np.mean(val_aps), np.std(val_aps)
    val_auc_mean, val_auc_std = np.mean(val_aucs), np.std(val_aucs)
    val_hit_mean, val_hit_std = np.mean(val_hits), np.std(val_hits)
    test_ap_mean, test_ap_std = np.mean(test_aps), np.std(test_aps)
    test_auc_mean, test_auc_std = np.mean(test_aucs), np.std(test_aucs)
    test_hit_mean, test_hit_std = np.mean(test_hits), np.std(test_hits)
    args.logger.info(f'[Final] model: {args.model}, attack: {args.attack} ({args.ptb_rate:.2f}), seeds: {args.seeds}, val_auc: {val_auc_mean:.4f}+-{val_auc_std:.4f}, val_hit: {val_hit_mean:.4f}+-{val_hit_std:.4f}, test_ap: {test_ap_mean:.4f}+-{test_ap_std:.4f}, test_auc: {test_auc_mean:.4f}+-{test_auc_std:.4f}, test_hit: {test_hit_mean:.4f}+-{test_hit_std:.4f}, neg_samples: {args.eval_neg_samples}')
    
    # save results in to a csv file
    row_data = {
        **vars(args),
        "AP": f"{np.mean(val_aps):.2%} ± {np.std(val_aps):.2%}",
        "AUC": f"{np.mean(val_aucs):.2%} ± {np.std(val_aucs):.2%}",
        "Hit": f"{np.mean(val_hits):.2%} ± {np.std(val_hits):.2%}",
        "Test_AP": f"{np.mean(test_aps):.2%} ± {np.std(test_aps):.2%}",
        "Test_AUC": f"{np.mean(test_aucs):.2%} ± {np.std(test_aucs):.2%}",
        "Test_Hit": f"{np.mean(test_hits):.2%} ± {np.std(test_hits):.2%}"
        }
    
    df = pd.DataFrame([row_data])
    current_time = datetime.now().strftime('%Y%m%d')
    id = args.id
    output_file = f"T-Spear_Attacks_results_neg_samples{args.eval_neg_samples}_{args.id}.csv"
    # df.to_csv(output_file, index=False)
    try:
        with open(output_file, 'x') as f:
            # File doesn't exist; create it and write data with header
            df.to_csv(f, index=False)
    except FileExistsError:
        # File exists; append without header
        df.to_csv(output_file, mode='a', header=False, index=False)
    


def attack_and_save(args):
    if args.attack == "none":
        print('returns nothing.')
        return

    data_dir = get_attacked_data_dir(args)
    
    # data_dir = '/'.join(data_dir.split('/')[:-2]) + '/' + 'C3_proposed_TGN_0.3_True_0'
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    node_feats, edge_feats, g, df = load_data(args.data, args)
    seed = 0
    set_random_seed(seed)

    if os.path.exists(f'{data_dir}/ext_full.pkl') and os.path.exists(f'{data_dir}/edges.csv'):
        print(f'[Complete] Already exists attacked dataset at {data_dir}, seed={seed}.\n')
        return

    ## Poisoning attack
    attacker = TemporalAttack(args)
    ptb_node_feats, ptb_edge_feats, ptb_g, ptb_df = attacker.attack(
        orig_node_feats=node_feats, 
        orig_edge_feats=edge_feats, 
        orig_g=g, 
        orig_df=df, 
        ptb_rate=args.ptb_rate, 
        args=args,
        seed=seed,
    )
    ## save attacked dataset
    print(ptb_edge_feats)
    # if args.data == 'wiki':
    if ptb_node_feats is not None:
        torch.save(ptb_node_feats, f'{data_dir}/node_features.pt')
    if ptb_edge_feats is not None:
        torch.save(ptb_edge_feats, f'{data_dir}/edge_features.pt')

    with open(f'{data_dir}/ext_full.pkl', 'wb') as f:
        pickle.dump(ptb_g, f)
    ptb_df.to_csv(f'{data_dir}/edges.csv', index=False)
    print(f'[Complete] Save attacked dataset at {data_dir}, seed={seed}.\n')


if __name__ == "__main__":
    start_time = time.time()
    args = parse_argument()
    if args.attack != "none":
        attack_and_save(args)
        train_main(args)
    else:
        train_main(args)
    end_time = time.time()
    print("Total time taken: ", end_time-start_time)
