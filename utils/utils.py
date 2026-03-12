import random
from tqdm import tqdm
import yaml
import logging
from functools import partial
import numpy as np
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.models import MetaPath2Vec
import torch
import torch.nn as nn
from torch import optim as optim

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class ContextualStructuralEncoder(torch.nn.Module):
    def __init__(self,
                 edge_index_dict,
                 metapath_schemas,
                 args
                 ):
        super(ContextualStructuralEncoder, self).__init__()
        
        self.cse_encoder = MetaPath2Vec(
                    edge_index_dict,
                    args.mps_embedding_dim,
                    metapath_schemas,
                    args.mps_walk_length,
                    args.mps_context_size,
                    args.mps_walks_per_node,
                    args.mps_num_negative_samples,
                    sparse=True
                )
        

    def forward(self, node_type):
        return self.cse_encoder(node_type)

    def loader(self, **kwargs):
        return self.cse_encoder.loader(**kwargs)

    def loss(self, pos_rw, neg_rw):
        return self.cse_encoder.loss(pos_rw, neg_rw)

def train_contextual_structural_encoder(args, cse_model, num_epochs, device):

    cse_model.to(device)
    cse_model.train()

    for epoch in range(num_epochs):
        walk_loader = cse_model.loader(batch_size=args.mps_batch_size, shuffle=True)
        optimizer = torch.optim.SparseAdam(list(cse_model.parameters()), lr=args.mps_lr)
        epoch_loss = 0.0
        for pos_walks, neg_walks in walk_loader:
            optimizer.zero_grad()
            loss = cse_model.loss(pos_walks.to(device), neg_walks.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return cse_model


def normalized_sum(embedding):
    # rw_embedding : num_nodes x num_dim
    # --> result.sum(axis=1) == 1
    return torch.matmul(torch.diag(1 / embedding.sum(axis=1)), embedding)


def normalized_unit_sphere(embedding):
    return torch.matmul(torch.diag(1 / embedding.sum(axis=1) ** 2), embedding)


# Standardize the feature
def standardize(embedding):
    mean = embedding.mean(dim=0).expand_as(embedding)
    std = torch.std(embedding, dim=0).expand_as(embedding)
    embedding = (embedding - mean) / (std + 1e-12)
    return embedding
