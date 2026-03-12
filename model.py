from itertools import chain
from functools import partial
import torch
import torch.nn as nn
import dgl
from dgl import DropEdge

import torch as pt
import torch_geometric as pyg
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import torch_geometric
import time
import argparse



from typing import List

import dgl
from encoder import GATConv
from utils import create_activation

import torch as pt
import torch_geometric as pyg
import GCL
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import warnings
import torch_geometric
import time
import argparse
    
def sample_nodes_for_dimensional_basis(sample_size, features, random=False) :
    with torch.no_grad():
        if random !=True :
            sampled_features = features[:sample_size, :]
        else:
            sampled_features = features[pt.randperm(features.shape[0]),:][:sample_size, :]
        return sampled_features
    
class DimensionAwareEncoder(nn.Module):
    def __init__(self, n_in, n_h, n_out, activator):
        super(DimensionAwareEncoder, self).__init__()
        self.act = activator()
        self.lin_in = nn.Linear(n_in, n_h)
        self.lin_h1 = nn.Linear(n_h, n_h)
        self.lin_out = nn.Linear(n_h, n_out)
        self.sample = []

    def encode(self, x):
        z = self.act(self.lin_in(x))
        z = self.act(self.lin_h1(z))
        return self.lin_out(z)

    def forward(self, x):
        self.sample = x
        self.out = self.encode(x.T)
        return self.out
         
    def dimensional_loss(self):
        return self.out.mean(dim=0).pow(2).mean()
    
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  
        beta = torch.softmax(w, dim=0) 
        beta = beta.expand((z.shape[0],) + beta.shape)  
        out_emb = (beta * z).sum(1) 
        att_mp = beta.mean(0).squeeze()

        return out_emb, att_mp


class HANLayer(nn.Module):
    def __init__(self, num_metapath, in_dim, out_dim, nhead,
                 feat_drop, attn_drop, negative_slope, residual, activation, norm, concat_out):
        super(HANLayer, self).__init__()
        self.gat_layers = GATConv(
                in_dim, out_dim, nhead,
                feat_drop, attn_drop, negative_slope, residual, activation, norm=norm, concat_out=concat_out)
        
        self.semantic_attention = SemanticAttention(in_size=out_dim * nhead)

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, new_g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers(new_g, h).flatten(1)) 
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1) 
        out, att_mp = self.semantic_attention(semantic_embeddings)

        return out, att_mp


class HAN(nn.Module):
    def __init__(self,
                 num_metapath,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False
                 ):
        super(HAN, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.han_layers = nn.ModuleList()
        self.activation = create_activation(activation)
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else create_activation(None)
        
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.han_layers.append(HANLayer(num_metapath,
                                            in_dim, out_dim, nhead_out,
                                            feat_drop, attn_drop, negative_slope, last_residual, last_activation,
                                            norm=last_norm, concat_out=concat_out))
        else:
            self.han_layers.append(HANLayer(num_metapath,
                                            in_dim, num_hidden, nhead,
                                            feat_drop, attn_drop, negative_slope, residual, self.activation, norm=norm,
                                            concat_out=concat_out))
            for l in range(1, num_layers - 1):
                self.han_layers.append(HANLayer(num_metapath,
                                                num_hidden * nhead, num_hidden, nhead,
                                                feat_drop, attn_drop, negative_slope, residual, self.activation,
                                                norm=norm, concat_out=concat_out))
            self.han_layers.append(HANLayer(num_metapath,
                                            num_hidden * nhead, out_dim, nhead_out,
                                            feat_drop, attn_drop, negative_slope, last_residual,
                                            activation=last_activation, norm=last_norm, concat_out=concat_out))

    def forward(self, gs: List[dgl.DGLGraph], h, return_hidden=False):
        for gnn in self.han_layers:
            h, att_mp = gnn(gs, h)
        return h, att_mp


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss
    

class MUG(nn.Module):
    def __init__(
            self,
            args,
            num_metapath,
            focused_feature_dim,
            Dimension_encoder,
            sample, 
            sample_size
    ):
        super(MUG, self).__init__()

        self.num_metapath = num_metapath
        self.focused_feature_dim = focused_feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.num_out_heads = args.num_out_heads
        self.activation = args.activation
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.negative_slope = args.negative_slope
        self.residual = args.residual
        self.norm = args.norm
        self.encoder_type = args.encoder
        self.decoder_type = args.decoder
        self.loss_fn = args.loss_fn
        self.enc_dec_input_dim = args.feature_signal_dim 
        self.losslam_scatter = args.losslam_scatter
        self.Dimension_encoder = Dimension_encoder
        self.sample = sample
        self.sample_size = sample_size
        self.d_sample_matrix = []
        
        
        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        if self.encoder_type in ("gat", "dotgat", "han"):
            enc_num_hidden = self.hidden_dim // self.num_heads
            enc_nhead = self.num_heads
        else:
            enc_num_hidden = self.hidden_dim
            enc_nhead = 1

        if self.decoder_type in ("gat", "dotgat", "han"):
            dec_num_hidden = self.hidden_dim // self.num_out_heads
            dec_nhead = self.num_out_heads
        else:
            dec_num_hidden = self.hidden_dim
            dec_nhead = 1
        dec_in_dim = self.hidden_dim

        self.encoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.encoder_type,
            enc_dec="encoding",
            in_dim=self.enc_dec_input_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=self.num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
        )

        self.decoder = setup_module(
            num_metapath=self.num_metapath,
            m_type=self.decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=self.enc_dec_input_dim,
            num_layers=1,
            nhead=enc_nhead,
            nhead_out=dec_nhead,
            activation=self.activation,
            dropout=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=True,
        )
        self.alpha_l = args.alpha_l
        self.__cache_gs = None
        self.mask = nn.Parameter(torch.zeros(1, self.enc_dec_input_dim))
        self.linear = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.mp_edge_recon_loss_weight = args.mp_edge_recon_loss_weight
        self.losslam_sig_cross = args.losslam_sig_cross
        self.mp_edge_mask_rate = args.mp_edge_mask_rate
        self.mp_edge_alpha_l = args.mp_edge_alpha_l
        self.mp_edge_recon_loss = self.setup_loss_fn(self.loss_fn, self.mp_edge_alpha_l)
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.unified_feature = args.unified_feature



    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "~" in input_mask_rate: 
                mask_rate = [float(i) for i in input_mask_rate.split('~')]
                assert len(mask_rate) == 2
                if get_min:
                    return mask_rate[0]
                else:
                    return torch.empty(1).uniform_(mask_rate[0], mask_rate[1]).item()
            elif "," in input_mask_rate: 
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(mask_rate) == 3
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def update_sample(self, x, if_rand=False):
        with torch.no_grad():
            self.d_sample_matrix = self.sample(self.sample_size, x, if_rand)
            
    def forward(self, feats, mps, **kwargs):  
        dimension_sig = self.Dimension_encoder(self.d_sample_matrix)
        
        if self.unified_feature:
            unified_feat = self.feature_sig_propagate(feats[0][:, :self.focused_feature_dim], dimension_sig)
        else:
            unified_feat = self.feature_sig_propagate(feats[0], dimension_sig)
        
        
        gs = self.mps_to_gs(mps)
        enc_out, _ = self.encoder(gs, unified_feat, return_hidden=False)
        
        edge_recon_loss = self.mask_mp_edge_reconstruction(unified_feat, mps, kwargs.get("epoch", None))
        loss = self.mp_edge_recon_loss_weight * edge_recon_loss
        loss_scatter = self.ssl_loss_fn_scatter(enc_out)
        loss += self.losslam_scatter * loss_scatter
        loss_sig = self.dim_loss_fn()
        loss += self.losslam_sig_cross * loss_sig
        
        return loss, loss.item(),loss_sig.item(),loss_scatter.item()
    
    def dim_loss_fn(self):
        return self.Dimension_encoder.dimensional_loss()
    
    def feature_sig_propagate(self, x, dimension_sig):
        return F.normalize(x @ dimension_sig)
    
    def ssl_loss_fn_scatter(self, z):
        z = F.normalize(z, dim=1)
        return z.mean(dim=0).pow(2).mean()
    

    def mask_mp_edge_reconstruction(self, feat, mps, epoch):
        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i]) 
        enc_rep, _ = self.encoder(masked_gs, feat, return_hidden=False)
        rep = self.encoder_to_decoder_edge_recon(enc_rep)

        if self.decoder_type == "mlp":
            feat_recon = self.decoder(rep)
        else:
            feat_recon, att_mp = self.decoder(masked_gs, rep)

        gs_recon = torch.mm(feat_recon, feat_recon.T)

        loss = None
        for i in range(len(mps)):
            if loss is None:
                loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
            else:
                loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
        return loss
  
    
    def get_embeds(self, feats, mps, if_rand=False, **kwargs):
        with torch.no_grad():
            self.eval()
            if self.unified_feature:
                original_attrs = feats[0][:, :self.focused_feature_dim]
            else:
                original_attrs = feats[0]
                
            gs = self.mps_to_gs(mps)
            self.update_sample(original_attrs, if_rand)
            basis_matrix = self.Dimension_encoder(self.d_sample_matrix)
            unified_features = self.feature_sig_propagate(original_attrs, basis_matrix)
            rep, _ = self.encoder(gs, unified_features)
            return rep.detach()


    def mps_to_gs(self, mps):
        if self.__cache_gs is None:
            gs = []
            for mp in mps:
                indices = mp._indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                gs.append(cur_graph)
            return gs
        else:
            return self.__cache_gs


def setup_module(m_type, num_metapath, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual,
                 norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "han":
        mod = HAN(
            num_metapath=num_metapath,
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=nn.BatchNorm1d,
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod
