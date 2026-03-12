import argparse

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


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--feat_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=5e-05,
                        help="learning rate")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")

    parser.add_argument("--encoder", type=str, default="han")
    parser.add_argument("--decoder", type=str, default="han")
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--alpha_l", type=float, default=2, help="pow index for sce loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--scheduler_gamma", type=float, default=0.99,
                        help="decay the lr by gamma for ExponentialLR scheduler")

    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--test_dataset', type=str, default='acm,dblp,aminer,freebase', help='acm,dblp,aminer,freebase')
    parser.add_argument('--ratio', type=int, default=[60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--eva_lr', type=float, default=0.01) 
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--l2_coef', type=float, default=0)

    parser.add_argument("--use_mp2vec_feat_pred", action="store_true")
    parser.add_argument("--mps_lr", type=float, default=0.005)
    parser.add_argument('--mps_embedding_dim', type=int, default=128)
    parser.add_argument('--mps_walk_length', type=int, default=5)
    parser.add_argument('--mps_context_size', type=int, default=3)
    parser.add_argument('--mps_walks_per_node', type=int, default=3)
    parser.add_argument('--mps_num_negative_samples', type=int, default=1)
    parser.add_argument('--mps_batch_size', type=int, default=128)
    parser.add_argument('--mps_epoch', type=int, default=20)
    parser.add_argument('--mp2vec_feat_pred_loss_weight', type=float, default=0.1)
    parser.add_argument("--mp2vec_feat_alpha_l", type=float, default=2)
    parser.add_argument("--mp2vec_feat_drop", type=float, default=.2)
    
    parser.add_argument("--use_cfg", action="store_true", help="Set to True to read config file")
    parser.add_argument("--unified_feature", type=str,default="True", help="unified_feature")
    parser.add_argument("--mp_edge_mask_rate", type=str, default="0.5,0.005,0.8")
    parser.add_argument("--mp_edge_alpha_l", type=float, default=2)

    parser.add_argument("--task", type=str, default="classification", choices=["classification", "clustering"])
    
    parser.add_argument('--sample_size', type=int, default=2048, help='feature sample batch size')
    parser.add_argument('--feature_signal_dim', type=int, default=2048, help='feature signal dim')
    parser.add_argument('--activator', type=str, default='PReLU', help='Activator name: PReLU, ReLU')
    parser.add_argument('--if_rand', type=str, default='True', help='feature sample if_rand: True, False')
    parser.add_argument('--losslam_sig_cross', type=float, default=200, help='hyper-parameter of sig_cross loss')
    parser.add_argument('--losslam_scatter', type=float, default=200, help='hyper-parameter of ssl loss')
    parser.add_argument('--mp_edge_recon_loss_weight', type=float, default=1)
    
    parser.add_argument('--log_dir', type=str, default='./log/', help='log')
    args, _ = parser.parse_known_args()
    for key, value in datasets_args[args.dataset].items():
        setattr(args, key, value)
    return args
