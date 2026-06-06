
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c

GIE_MODELS = ["GIE"]


class BaseH(KGModel):

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1= nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2



class GIE(BaseH):

    def __init__(self, args):
        super(GIE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def _get_curvature(self, c_param, rel_idx):
        if self.multi_c:
            return F.softplus(c_param[rel_idx])
        return F.softplus(c_param).expand(rel_idx.shape[0], -1)

    def get_queries(self, queries):

        # import pdb
        # pdb.set_trace()

        rel_idx = queries[:, 1]
        c1 = self._get_curvature(self.c1, rel_idx)
        head1 = expmap0(self.entity(queries[:, 0]), c1)
        rel1, rel2 = torch.chunk(self.rel(rel_idx), 2, dim=1)
        rel1 = expmap0(rel1, c1)
        rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(rel_idx), lhs)
        c2 = self._get_curvature(self.c2, rel_idx)
        head2 = expmap0(self.entity(queries[:, 0]), c2)
        rel1, rel2 = torch.chunk(self.rel(rel_idx), 2, dim=1)
        rel11 = expmap0(rel1, c2)
        rel21= expmap0(rel2, c2)
        lhss = project(mobius_add(head2, rel11, c2), c2)
        res11 = givens_rotations(self.rel_diag2(rel_idx), lhss)
        c = self._get_curvature(self.c, rel_idx)
        head = self.entity(queries[:, 0])
        rot_mat, _ = torch.chunk(self.rel_diag(rel_idx), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([res1.view(-1, 1, self.rank),res11.view(-1, 1, self.rank),rot_q], dim=1)
        context_vec = self.context_vec(rel_idx).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(rel_idx), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])
