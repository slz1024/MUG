import warnings
datasets_args = {
    "dblp": {
        "type_num": [4057, 14328, 7723, 20],  # the number of every node type
        "nei_num": 1,  # the number of neighbors' types
        "n_labels": 4,
    },
    "aminer": {
        "type_num": [6564, 13329, 35890],
        "nei_num": 2,
        "n_labels": 4,
    },
    "freebase": {
        "type_num": [3492, 2502, 33401, 4459],
        "nei_num": 3,
        "n_labels": 3,
    },
    "acm": {
        "type_num": [4019, 7167, 60],
        "nei_num": 2,
        "n_labels": 3,
    },
}
warnings.filterwarnings('ignore')

import datetime
import sys
import warnings
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.nn.models import MetaPath2Vec
import torch as pt
import torch.nn as nn
import torch_geometric as pyg
from tqdm import tqdm
import time
import argparse

from model import MUG,DimensionAwareEncoder,sample_nodes_for_dimensional_basis,setup_module
from utils import (evaluate, load_best_configs, load_data,ContextualStructuralEncoder, train_contextual_structural_encoder,
                         preprocess_features,
                         set_random_seed,create_activation)
from utils.params import build_args

warnings.filterwarnings('ignore')


def main(args):
    set_random_seed(args.seed)
    log_dir = args.log_dir+'log_'+str(args.dataset)+'.txt'
    with open(log_dir, 'a') as f:
        f.write('\n\n\n')
        f.write(str(args))
    
    (nei_index, feats, mps, pos, _, _, _, _), g, processed_metapaths = \
        load_data(args.dataset, args.ratio, args.type_num)

    print(type(g))
    print(list(g.edge_index_dict.keys()))
    feats_dim_list = [i.shape[1] for i in feats]

    num_mp = int(len(mps))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", num_mp)
    if args.unified_feature:
        cse_model = ContextualStructuralEncoder(
            g.edge_index_dict,
            processed_metapaths,
            args
        )
        cse_model = train_contextual_structural_encoder(args, cse_model, args.mps_epoch, args.device)
        z_struct = cse_model('target').detach()

        del cse_model
        if args.device.type == 'cuda':
            z_struct = z_struct.cpu()
            torch.cuda.empty_cache()
        feats[0] = torch.hstack([feats[0], torch.FloatTensor(preprocess_features(z_struct))])
    
    z_dim = feats_dim_list[0]
    feature_dim = feats_dim_list[0]
    activator = nn.PReLU if args.activator == 'PReLU' else nn.ReLU
    dimension_encoder = DimensionAwareEncoder(args.sample_size, args.feature_signal_dim*2, args.feature_signal_dim, activator)
    model = MUG(args, num_mp, feature_dim,dimension_encoder,sample_nodes_for_dimensional_basis,args.sample_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    if args.scheduler:
        print("--- Use schedular ---")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    model.to(args.device)
    feats = [feat.to(args.device) for feat in feats]
    mps = [mp.to(args.device) for mp in mps]
    

    best = 1e9
    starttime = datetime.datetime.now()
    best_model_state_dict = None
    
    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(args.epochs):
            model.update_sample(feats[0][:, :z_dim], if_rand=args.if_rand)
            model.train()
            optimizer.zero_grad()
            loss, loss_item,loss_sig,loss_scatter = model(feats, mps, nei_index=nei_index, epoch=epoch)
            
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            pbar.set_postfix({'loss': loss.item()})
            pbar.update()

    model.eval()
    
    
    
    if args.task == 'classification':
        test_datasets = args.test_dataset.split(',')
        for t in range(len(test_datasets)):
            macro_score_list, micro_score_list, auc_score_list = [], [], []
            (_, feats, mps, pos, label, idx_train, idx_val, idx_test), g, processed_metapaths = \
            load_data(test_datasets[t], args.ratio, datasets_args[test_datasets[t]]["type_num"])
            label = label.to(args.device)
            idx_train = [i.to(args.device) for i in idx_train]
            idx_val = [i.to(args.device) for i in idx_val]
            idx_test = [i.to(args.device) for i in idx_test]
            feats = [feat.to(args.device) for feat in feats]
            mps = [mp.to(args.device) for mp in mps]
            
            nb_classes = label.shape[-1]
            embeds = model.get_embeds(feats, mps)
            
            for i in range(len(idx_train)):
                macro_score, micro_score, auc_score = evaluate(embeds, idx_train[i], idx_val[i], idx_test[i], label, nb_classes, args.device,
                                                               args.eva_lr, args.eva_wd,log_dir,test_datasets[t],args.ratio[i])
                macro_score_list.append(macro_score)
                micro_score_list.append(micro_score)
                auc_score_list.append(auc_score)
    
    else:
        sys.exit('wrong args.task.')

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    

if __name__ == "__main__":
    args = build_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        args.device = torch.device("cpu")
    config_file_name = "configs.yml"
    args = load_best_configs(args, config_file_name)
    for trial in range(10):
        main(args)