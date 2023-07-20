import itertools
from functools import lru_cache
from typing import Tuple

import torch
from torch import nn


def LinearSolverSparse():
    class LinearSolverSparse_(torch.autograd.Function):
        @staticmethod
        @torch.jit.script
        def solve_system(B, K, wc, wr, max_iters: int, bs: int, c: int, h: int, w: int):
            eps = torch.tensor([1e-6 * B.size()[0] * B.size()[1] * B.size()[2]])
            eps = (eps * eps).type_as(B)

            x = B
            xx = x.movedim(-1, 1).view(bs, c, h, w)
            r = B - (K * xx +
                     torch.functional.F.pad(wc * xx[..., :-1, :], (0, 0, 1, 0), 'constant', 0.) +
                     torch.functional.F.pad(wc * xx[..., 1:, :], (0, 0, 0, 1), 'constant', 0.) +
                     torch.functional.F.pad(wr * xx[..., :-1], (1, 0, 0, 0), 'constant', 0.) +
                     torch.functional.F.pad(wr * xx[..., 1:], (0, 1, 0, 0), 'constant', 0.)).movedim(1, -1).view(bs, h * w, c)
            p = r

            rr = torch.sum(r * r, dim=-2)

            for i in range(max_iters):
                pp = p.movedim(-1, 1).view(bs, c, h, w)
                Ap = (K * pp +
                      torch.functional.F.pad(wc * pp[..., :-1, :], (0, 0, 1, 0), 'constant', 0.) +
                      torch.functional.F.pad(wc * pp[..., 1:, :], (0, 0, 0, 1), 'constant', 0.) +
                      torch.functional.F.pad(wr * pp[..., :-1], (1, 0, 0, 0), 'constant', 0.) +
                      torch.functional.F.pad(wr * pp[..., 1:], (0, 1, 0, 0), 'constant', 0.)).movedim(1, -1).view(bs, h * w, c)
                a = (rr / torch.sum(p * Ap, dim=-2)).nan_to_num().unsqueeze(1).expand_as(x)
                x = x + a * p
                r_new = r - a * Ap

                rr_new = torch.sum(r_new * r_new, dim=-2)
                if (rr_new.sum() < eps).all():
                    break

                b = (torch.sum(r_new * r_new, dim=-2) / rr).nan_to_num().unsqueeze(1).expand_as(x)
                p = r_new + b * p
                r = r_new
                rr = rr_new

            return x

        @staticmethod
        @torch.jit.script
        def make_sparse_helper(indices, x, gradB, bs: int, m: int):
            indices_ = indices[1:, :indices.size(-1) // bs]
            gradA = torch.sparse_coo_tensor(indices, -torch.sum((gradB[:, indices_[0]] * x[:, indices_[1]]), -1).flatten(0, 1),
                                            (bs, m, m))
            return gradA

        @staticmethod
        def forward(ctx, A, B, max_iters, K, wc, wr, sizes):
            bs, c, h, w = sizes

            K = K.unsqueeze(1).expand(-1, c, -1, -1)
            wc = wc.unsqueeze(1).expand(-1, c, -1, -1)
            wr = wr.unsqueeze(1).expand(-1, c, -1, -1)

            x = LinearSolverSparse_.solve_system(B, K, wc, wr, max_iters, bs, c, h, w)

            ctx.save_for_backward(A._indices(), B, x, K, wc, wr)
            ctx.max_iters = max_iters
            ctx.sizes = sizes

            return x

        @staticmethod
        def backward(ctx, grad):
            indices, B, x, K, wc, wr = ctx.saved_tensors
            bs, c, h, w = ctx.sizes

            gradB = LinearSolverSparse_.solve_system(grad, K, wc, wr, ctx.max_iters, bs, c, h, w)

            gradA = LinearSolverSparse_.make_sparse_helper(indices, x, gradB, bs, h * w)

            return gradA, gradB, None, None, None, None, None

    return LinearSolverSparse_.apply


class SoftGraphCut(nn.Module):
    def __init__(self, max_iters=500):
        super().__init__()

        self.max_iters = max_iters

        self.liner_system_solver = LinearSolverSparse()

        self.pad_col_before = nn.ZeroPad2d((0, 0, 1, 0))
        self.pad_col_after = nn.ZeroPad2d((0, 0, 0, 1))
        self.pad_row_before = nn.ZeroPad2d((1, 0, 0, 0))
        self.pad_row_after = nn.ZeroPad2d((0, 1, 0, 0))

    @lru_cache(maxsize=None)
    def indices_from_batch_size(self,s0, s1, bs, device):
        
        indices = []
        # indices for K coefficients
        for i in range(s0):
            for j in range(s1):
                target_index = j + i * s1
                indices.append([target_index, target_index])
        # indices for column coefficients
        for i in range(s0 - 1):
            for j in range(s1):
                target_index = j + (i + 1) * s1
                coef_index = j + i * s1
                indices.append([target_index, coef_index])
        for i in range(s0 - 1):
            for j in range(s1):
                target_index = j + i * s1
                coef_index = j + (i + 1) * s1
                indices.append([target_index, coef_index])
        # indices for row coefficients
        for i in range(s0):
            for j in range(s1 - 1):
                target_index = j + 1 + i * s1
                coef_index = j + i * s1
                indices.append([target_index, coef_index])
        for i in range(s0):
            for j in range(s1 - 1):
                target_index = j + i * s1
                coef_index = j + 1 + i * s1
                indices.append([target_index, coef_index])
        
        ret = [[[b] + i for i in indices] for b in range(bs)]
        ret = list(itertools.chain(*ret))
        return torch.tensor(list(zip(*ret)), device=device)

    @staticmethod
    @torch.jit.script
    def create_sparse_matrix_helper(values, indices, size: Tuple[int, int], bs: int):
        return torch.sparse_coo_tensor(indices, values, (bs, size[0] * size[1], size[0] * size[1]))

    def forward(self, pixel_scores, weights_col, weights_row):
        bs = pixel_scores.size(0)
        pixel_scores_size = tuple(pixel_scores.size())
        assert len(weights_col.size()) == 3
        assert len(weights_row.size()) == 3
        assert bs == weights_col.size(0)
        assert bs == weights_row.size(0)
        assert pixel_scores_size[-2] == weights_col.size(1) + 1
        assert pixel_scores_size[-1] == weights_col.size(2)
        assert pixel_scores_size[-2] == weights_row.size(1)
        assert pixel_scores_size[-1] == weights_row.size(2) + 1

        B = torch.flatten(pixel_scores, -2, -1).swapaxes(1, -1)

        K = torch.ones_like(pixel_scores[:, 0, ...])

        weights_col_pad_before = self.pad_col_before(weights_col)
        weights_col_pad_after = self.pad_col_after(weights_col)
        weights_row_pad_before = self.pad_row_before(weights_row)
        weights_row_pad_after = self.pad_row_after(weights_row)

        K = K + weights_col_pad_before + weights_col_pad_after + weights_row_pad_before + weights_row_pad_after

        weights_col_coeff = -weights_col
        weights_row_coeff = -weights_row

        values = torch.cat((K.flatten(-2, -1),
                            weights_col_coeff.flatten(-2, -1), weights_col_coeff.flatten(-2, -1),
                            weights_row_coeff.flatten(-2, -1), weights_row_coeff.flatten(-2, -1)), dim=1)
        values = values.flatten(0, 1)

        indices = self.indices_from_batch_size(pixel_scores_size[-2], pixel_scores_size[-1] ,bs, str(B.device))

        A = SoftGraphCut.create_sparse_matrix_helper(values, indices, pixel_scores_size[-2:], bs)

        x = self.liner_system_solver(A, B, self.max_iters, K, weights_col_coeff, weights_row_coeff, pixel_scores_size)
        return x.type_as(pixel_scores).swapaxes(1, -1).view(pixel_scores_size)
