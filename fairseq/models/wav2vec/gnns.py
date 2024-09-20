import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fairseq.modules import GradMultiply
from torch_geometric.nn.dense import DenseGCNConv, DenseGraphConv, DenseSAGEConv, DenseGATConv
# from torch.nn import Linear
from torch import Tensor
from typing import Optional
from torch_geometric.nn import Linear, BatchNorm

class CustomDenseGraphConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GraphConv`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
    ):
        assert aggr in ['add', 'mean', 'max']
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.lin_rel = Linear(in_channels, out_channels, bias=bias, weight_initializer='glorot')
        self.lin_root = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        # self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, C = x.size()

        if self.aggr == 'add':
            out = torch.matmul(adj, x)
        elif self.aggr == 'mean':
            out = torch.matmul(adj, x)
            out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1)
        elif self.aggr == 'max':
            out = x.unsqueeze(-2).repeat(1, 1, N, 1)
            adj = adj.unsqueeze(-1).expand(B, N, N, C)
            out[adj == 0] = float('-inf')
            out = out.max(dim=-3)[0]
            out[out == float('-inf')] = 0.
        else:
            raise NotImplementedError

        out = self.lin_rel(out)
        out = out + self.lin_root(x)

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    from: https://github.com/ReML-AI/MGTN/blob/76bf9ea1f036eec2374576f1d7509f8a2c5dd065/gnn.py#L7
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        # output = torch.matmul(adj, support)

        adj = adj.clone()
        idx = torch.arange(adj.shape[1], dtype=torch.long, device=adj.device)
        adj[:, idx, idx] = 2

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        output = torch.einsum('bnn,bnh->bnh',adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class custom_GCN(nn.Module):
    def __init__(self, nfeat, hidden_dim, nclass, dropout=0.5, grad_mul=1.0):
        super(custom_GCN, self).__init__()
        self.dropout = dropout
        self.nfeat = nfeat
        # GraphConvolution, GCNConv, GCNConv2
        graph_arch = CustomDenseGraphConv # GraphConvolution
        # self.gc1 = graph_arch(nfeat, int(hidden_dim/2))  # (feature_dim, node_dims)
        # self.gc2 = graph_arch(int(hidden_dim/2), hidden_dim)
        # self.gc3 = graph_arch(hidden_dim, nclass)
        self.gc1 = graph_arch(nfeat, hidden_dim)  # (feature_dim, node_dims)
        self.gc2 = graph_arch(hidden_dim, hidden_dim)
        self.gc3 = graph_arch(hidden_dim, nclass)

        # self.gc1 = GraphConvolution(nfeat, hidden_dim)  # (feature_dim, node_dims)
        # self.gc2 = GraphConvolution(hidden_dim, nclass)
        self.relu = nn.LeakyReLU(0.2)
        # self.layer_norm1 = nn.LayerNorm(hidden_dim)
        # self.layer_norm2 = nn.LayerNorm(nclass)
        # self.bn0 = torch.nn.BatchNorm1d(nfeat)
        # self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        # self.bn3 = torch.nn.BatchNorm1d(nclass)
        # self.bn0 = torch.nn.BatchNorm1d(39)
        # self.bn1 = torch.nn.BatchNorm1d(39)
        # self.bn2 = torch.nn.BatchNorm1d(39)

        self.grad_mul = grad_mul

    def forward(self, x, adj=None):
        # x: Node feature matrix of shape [batch_size, num_nodes, feature_dim]
        if adj is None:
            # if self.first:
            #     for b in range(x.shape[0]):
            #         adj_mat =  torch.corrcoef(x[b]).detach().cpu().numpy()
            #         import matplotlib.pyplot as plt
            #         plt.imshow(adj_mat, cmap='hot')
            #         plt.savefig(f'{b}.png')
            temp_adj = []
            for b in range(x.shape[0]):
                if padding_mask[b].any():
                    try:
                        temp_adj.append(torch.corrcoef(x[b,:, :-int(padding_mask[b].sum().item()/x.shape[1])]))
                    except Exception as e:
                        import pdb
                        pdb.set_trace()
                else:
                    temp_adj.append(torch.corrcoef(x[b]))
            adj = torch.stack(temp_adj)
            # adj = torch.stack([torch.corrcoef(x[b,:, :padding_mask[b].sum().item()/x.shape[1] if padding_mask[b].any() else ]) for b in range(x.shape[0])])
            # print(adj)
            adj = (adj + 1.0) * 0.5 # From (-1, 1) to (0, 1)
            adj = torch.nan_to_num(adj, nan=0.0) #NOTE: TODO here
        adj = (adj + 1.0) * 0.5 # From (-1, 1) to (0, 1)
        adj = torch.nan_to_num(adj, nan=0.0) #NOTE: TODO here
            # adj = torch.where(adj > 0.8, adj, 0.0)
        # x = x[...,:self.nfeat]
        # x = F.pad(x, (0, self.nfeat-x.shape[-1]), value=0)
        # import pdb
        # pdb.set_trace()
        # x = self.bn0(x.transpose(1,2)).transpose(1,2)
        # x = self.bn0(x)
        x_ = self.gc1(x, adj)
        x_ = self.relu(x_)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        # x_ = self.bn1(x_.transpose(1,2)).transpose(1,2)
        # x_ = self.bn1(x_)
        x_ = self.gc2(x_, adj)
        x_ = self.relu(x_)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        # x_ = self.bn2(x_.transpose(1,2)).transpose(1,2)
        # x_ = self.bn2(x_)
        x_ = self.gc3(x_, adj)
        x_ = self.relu(x_)

        # x_ = self.gc2(x_, adj)
        # x_ = self.relu(x_)
        #x_ = F.log_softmax(x_, dim=1)
        #x_ = x_.transpose(0, 1)
        x_ = torch.max(x_, dim=1, keepdim=False)[0]
        # x_ = self.bn3(x_)

        # x_ = GradMultiply.apply(x_, 10)
        x_ = GradMultiply.apply(x_, self.grad_mul)
        return x_


class GCN(nn.Module):
    def __init__(self, nfeat, hidden_dim, nclass, dropout=0.5, grad_mul=1.0, typeFeature=2):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.nfeat = nfeat
        # GraphConvolution, GCNConv, GCNConv2
        graph_arch = DenseGraphConv # GraphConvolution
        # self.gc1 = graph_arch(nfeat, int(hidden_dim/2))  # (feature_dim, node_dims)
        # self.gc2 = graph_arch(int(hidden_dim/2), hidden_dim)
        # self.gc3 = graph_arch(hidden_dim, nclass)
        self.gc1 = graph_arch(nfeat, hidden_dim)  # (feature_dim, node_dims)
        self.gc2 = graph_arch(hidden_dim, hidden_dim)
        self.gc3 = graph_arch(hidden_dim, nclass)

        # self.gc1 = GraphConvolution(nfeat, hidden_dim)  # (feature_dim, node_dims)
        # self.gc2 = GraphConvolution(hidden_dim, nclass)
        self.relu = nn.LeakyReLU(0.2)
        # self.layer_norm1 = nn.LayerNorm(hidden_dim)
        # self.layer_norm2 = nn.LayerNorm(nclass)
        # self.bn0 = torch.nn.BatchNorm1d(nfeat)
        # self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        # self.bn3 = torch.nn.BatchNorm1d(nclass)
        # self.bn0 = torch.nn.BatchNorm1d(39)
        # self.bn1 = torch.nn.BatchNorm1d(39)
        # self.bn2 = torch.nn.BatchNorm1d(39)

        self.grad_mul = grad_mul
        self.typeFeature = typeFeature
        if self.typeFeature == 2:
            self.gc4 = graph_arch(nclass, nclass)

    def gnnBlock(self, x, adj, layer=3):
        x_ = self.gc1(x, adj)
        x_ = self.relu(x_)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        # x_ = self.bn1(x_.transpose(1,2)).transpose(1,2)
        # x_ = self.bn1(x_)
        x_ = self.gc2(x_, adj)
        x_ = self.relu(x_)
        x_ = F.dropout(x_, self.dropout, training=self.training)
        # x_ = self.bn2(x_.transpose(1,2)).transpose(1,2)
        # x_ = self.bn2(x_)
        x_ = self.gc3(x_, adj)
        x_ = self.relu(x_)
        return x_

    def forward(self, x, adj=None):
        # x: Node feature matrix of shape [batch_size, num_nodes, feature_dim]
        batch_size, frequency_band, time = x.shape
        if 1: #adj is None:
            # if self.first:
            #     for b in range(x.shape[0]):
            #         adj_mat =  torch.corrcoef(x[b]).detach().cpu().numpy()
            #         import matplotlib.pyplot as plt
            #         plt.imshow(adj_mat, cmap='hot')
            #         plt.savefig(f'{b}.png')
            temp_adj = []
            intra_adj = []
            print('--------------------GCN')
            for bs in range(batch_size):
                #if padding_mask[bs].any():
                #    try:
                #        x[bs] = x[bs,:, :-int(padding_mask[bs].sum().item()/x.shape[1])]
                #    except Exception as e:
                #        import pdb
                #        pdb.set_trace()

                # split by n nodes
                node = frequency_band
                overlap = 0.5 if self.typeFeature == 6 else 0
                step = int(node*(1-min(abs(overlap), 0.9)))
                f = []
                for t in range(0, time-node, step):
                    f.append(x[bs, :, t:t+node])

                if self.typeFeature in [1, 2]:
                    f = torch.stack([torch.corrcoef(fi.T) for fi in f])  # (n, node, node)
                    temp_adj.append(f)
                elif self.typeFeature in [3, 6, 7]:  # concat
                    if self.typeFeature == 7:
                        intra_adj.append(torch.stack([torch.corrcoef(fi.T) for fi in f]))
                    f = torch.stack(f, axis=0).reshape(-1, node)  # (num_step*frequency_band, node)
                    temp_adj.append(torch.corrcoef(f.T))
                elif self.typeFeature == 4:  # mean
                    f = torch.mean(torch.stack(f), dim=0)  # (frequency_band, node)
                    temp_adj.append(torch.corrcoef(f.T))
                elif self.typeFeature in [5, 8]:  # max
                    if self.typeFeature == 8:
                        intra_adj.append(torch.stack([torch.corrcoef(fi.T) for fi in f]))
                    f = torch.max(torch.stack(f), dim=0)[0]  # (frequency_band, node)
                    temp_adj.append(torch.corrcoef(f.T))
                else:
                    temp_adj.append(torch.corrcoef(x[bs]))  # (node, time)
            adj = torch.stack(temp_adj)
            intra_adj = [] if len(intra_adj) == 0 else torch.stack(intra_adj)
        adj = (adj + 1.0) * 0.5 # From (-1, 1) to (0, 1)
        adj = torch.nan_to_num(adj, nan=0.0) #NOTE: TODO here
            # adj = torch.where(adj > 0.8, adj, 0.0)
        # x = x[...,:self.nfeat]
        # x = F.pad(x, (0, self.nfeat-x.shape[-1]), value=0)
        # import pdb
        # pdb.set_trace()
        # x = self.bn0(x.transpose(1,2)).transpose(1,2)
        # x = self.bn0(x)
        if self.typeFeature in [1, 2]:
            xa = []
            for a in range(adj.shape[1]):
                x_ = self.gnnBlock(x, adj[:, a, :, :])
                xa.append(x_)
            xa = torch.stack(xa, axis=1)
            if self.typeFeature == 2:
                xa = torch.mean(xa, axis=2)
                adj2 = torch.stack([torch.corrcoef(a) for a in xa])
                x_ = self.gc4(xa, adj2)
                x_ = self.relu(x_)
                #x_ = F.dropout(x_, self.dropout, training=self.training)
                #x_ = self.gc4(x_, adj2)
                #x_ = self.relu(x_)
            else:
                x_ = torch.mean(xa, axis=1)
        else:
            x_ = self.gnnBlock(x, adj)
            if self.typeFeature in [7, 8]:
                xa = []
                for a in range(intra_adj.shape[1]):
                    xa.append(self.gnnBlock(x, intra_adj[:, a, :, :]))
                #xa = torch.mean(torch.stack(xa, axis=1), axis=1)
                xa = torch.stack(xa, axis=1)
                x_ = torch.unsqueeze(x_, dim=1)
                x_ = torch.cat([x_, xa], axis=1)
                x_ = torch.mean(x_, axis=1)
        # x_ = self.gc2(x_, adj)
        # x_ = self.relu(x_)
        #x_ = F.log_softmax(x_, dim=1)
        #x_ = x_.transpose(0, 1)
        x_ = torch.max(x_, dim=1, keepdim=False)[0]
        # x_ = self.bn3(x_)

        # x_ = GradMultiply.apply(x_, 10)
        x_ = GradMultiply.apply(x_, self.grad_mul)
        return x_


class GCN_v2(nn.Module):
    def __init__(self, nfeat, hidden_dim, nclass, dropout=0.5, grad_mul=1.0, n_hidden_dim=1, adj_thresh=-1.0):
        super(GCN_v2, self).__init__()
        self.dropout = dropout
        self.nfeat = nfeat

        # GraphConvolution, GCNConv, GCNConv2
        graph_arch = DenseGraphConv # GraphConvolution
        self.layers = nn.Sequential(
            graph_arch(nfeat, hidden_dim),
        )
        for i in range(n_hidden_dim):
            self.layers.append(
                graph_arch(hidden_dim, hidden_dim),
            )
        
        self.layers.append(
            graph_arch(hidden_dim, nclass),
        )

        self.relu = nn.LeakyReLU(0.2)
        self.grad_mul = grad_mul
        self.adj_thresh = adj_thresh

    def forward(self, x, adj=None):
        # x: Node feature matrix of shape [batch_size, num_nodes, feature_dim]
        if adj is None:
            # if self.first:
            #     for b in range(x.shape[0]):
            #         adj_mat =  torch.corrcoef(x[b]).detach().cpu().numpy()
            #         import matplotlib.pyplot as plt
            #         plt.imshow(adj_mat, cmap='hot')
            #         plt.savefig(f'{b}.png')
            temp_adj = []
            for b in range(x.shape[0]):
                if padding_mask[b].any():
                    try:
                        temp_adj.append(torch.corrcoef(x[b,:, :-int(padding_mask[b].sum().item()/x.shape[1])]))
                    except Exception as e:
                        import pdb
                        pdb.set_trace()
                else:
                    temp_adj.append(torch.corrcoef(x[b]))
            adj = torch.stack(temp_adj)
            # adj = torch.stack([torch.corrcoef(x[b,:, :padding_mask[b].sum().item()/x.shape[1] if padding_mask[b].any() else ]) for b in range(x.shape[0])])
            # print(adj)
            # adj = (adj + 1.0) * 0.5 # From (-1, 1) to (0, 1)
            # adj = torch.nan_to_num(adj, nan=0.0) #NOTE: TODO here
        adj = (adj + 1.0) * 0.5 # From (-1, 1) to (0, 1)
        adj = torch.nan_to_num(adj, nan=0.0) #NOTE: TODO here
        if self.adj_thresh != -1.0:
            adj = torch.where(adj > self.adj_thresh, adj, 0.0)

        for i in range(len(self.layers)):
            x = self.layers[i](x, adj)
            x = self.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = torch.max(x, dim=1, keepdim=False)[0]

        x = GradMultiply.apply(x, self.grad_mul)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, hidden_dim, nclass, dropout=0.5, grad_mul=1.0, n_hidden_dim=1, adj_thresh=-1.0, inner_dropout=0.0, add_self_loops=True, n_head=1, batch_norm=False, act='elu'):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nfeat = nfeat

        # GraphConvolution, GCNConv, GCNConv2
        graph_arch = DenseGATConv # GraphConvolution
        self.layers = nn.Sequential(
            graph_arch(nfeat, hidden_dim, dropout=inner_dropout, heads=n_head, concat=False if n_head!=1 else True),
        )

        self.use_batch_norm = batch_norm

        if self.use_batch_norm:
            self.bns = nn.Sequential(BatchNorm(hidden_dim))
        for i in range(n_hidden_dim):
            self.layers.append(
                graph_arch(hidden_dim, hidden_dim, dropout=inner_dropout, heads=n_head, concat=False if n_head!=1 else True),
            )
            if self.use_batch_norm:
                self.bns.append(BatchNorm(hidden_dim))

        self.layers.append(
            graph_arch(hidden_dim, nclass, dropout=inner_dropout, heads=n_head, concat=False if n_head!=1 else True),
        )
        if self.use_batch_norm:
            self.bns.append(BatchNorm(nclass))

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU() #nn.ReLU()
        else:
            self.act = nn.ELU()
        self.grad_mul = grad_mul
        self.adj_thresh = adj_thresh
        self.add_loop=add_self_loops

    def forward(self, x, adj=None, typeFeature=0):
        # x: Node feature matrix of shape [batch_size, num_nodes, feature_dim]
        batch_size, frequency_band, time = x.shape
        if adj is None:
            # if self.first:
            #     for b in range(x.shape[0]):
            #         adj_mat =  torch.corrcoef(x[b]).detach().cpu().numpy()
            #         import matplotlib.pyplot as plt
            #         plt.imshow(adj_mat, cmap='hot')
            #         plt.savefig(f'{b}.png')
            temp_adj = []
            print('--------------------GAT')
            for bs in range(batch_size):
                if padding_mask[bs].any():
                    try:
                        x[bs] = x[bs,:, :-int(padding_mask[bs].sum().item()/x.shape[1])]
                    except Exception as e:
                        import pdb
                        pdb.set_trace()

                if typeFeature in [1, 2]:
                    adjs = []
                    # split by n parts
                    n = 8
                    step = int(time/n)
                    # overlap by factor o
                    o = 0.5
                    for t in range(0, time-int(time*o/n), step):
                        adjs.append(torch.corrcoef(x[bs, :, t:t+int(time*o/n)].T))
                    temp_adj.append(torch.stack(adjs))

                    if typeFeature == 2:
                        temp_adj = torch.cat(temp_adj, axis=1)
                elif typeFeature == 3:
                    # split by n nodes
                    node = frequency_band
                    step = int(time/node)
                    f = []
                    for t in range(0, time-step, step):
                        f.append(x[bs, :, t:t+step])
                    f = torch.stack(f, axis=1)
                    temp_adj.append(torch.corrcoef(f.T))
                else:
                    temp_adj.append(torch.corrcoef(x[bs]))
            adj = torch.stack(temp_adj)
            # adj = torch.stack([torch.corrcoef(x[b,:, :padding_mask[b].sum().item()/x.shape[1] if padding_mask[b].any() else ]) for b in range(x.shape[0])])
            # print(adj)
            # adj = (adj + 1.0) * 0.5 # From (-1, 1) to (0, 1)
            # adj = torch.nan_to_num(adj, nan=0.0) #NOTE: TODO here
        adj = (adj + 1.0) * 0.5 # From (-1, 1) to (0, 1)
        adj = torch.nan_to_num(adj, nan=0.0) #NOTE: TODO here
        x = torch.nan_to_num(x, nan=0.0)

        if self.adj_thresh != -1.0:
            adj = torch.where(adj > self.adj_thresh, adj, 0.0)

        for i in range(len(self.layers)):
            if typeFeature == 1:
                xa = []
                for a in range(adj.shape[1]):
                    x_ = F.dropout(x, self.dropout, training=self.training)
                    x_ = self.layers[i](x_, adj[:, a, :, :], add_loop=self.add_loop)
                    if i != len(self.layers) - 1:
                        x_ = self.act(x_)
                    xa.append(x_)
                x = torch.mean(torch.stack(xa), axis=1)
            else:
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.layers[i](x, adj, add_loop=self.add_loop)
                # import pdb; pdb.set_trace()
                if self.use_batch_norm:
                    x = self.bns[i](x.view(-1, x.size(-1))).view(x.size())
                if i != len(self.layers) - 1:
                    x = self.act(x)

        x = torch.max(x, dim=1, keepdim=False)[0]

        x = GradMultiply.apply(x, self.grad_mul)
        return x
