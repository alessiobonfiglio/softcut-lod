import itertools
from functools import lru_cache

import torch
from torch import multiprocessing
from torch import nn
from torch_sparse_solve_cpp import _klu_solve, _coo_to_csc


def batch_process(convert_matrix, num_workers=8, multiprocess=False, multithread=False):
    assert (multiprocess and multithread) is False, \
        "Either multiprocess or multithread can be True, not both"

    if multiprocess:
        pool = multiprocessing.get_context('forkserver').Pool(num_workers)
    else:
        pool = None

    def decorator(single_sample_fn):
        if convert_matrix:
            def wrapper(m, Ax, Bflat, Ai, Aj):
                batch_size = Ax.size(0)
                device = Ax.device
                dtype = Ax.dtype

                Ax = Ax.to('cpu',torch.float64, non_blocking=True)
                Bflat = Bflat.to('cpu',torch.float64, non_blocking=True)
                Ai = Ai.to('cpu', non_blocking=True)
                Aj = Aj.to('cpu', non_blocking=True)

                if batch_size == 1:
                    _, Ap, Ai, Ax = single_sample_fn(m, Ax[0], Bflat[0], Ai[0], Aj[0])
                    out = Bflat.to(device, dtype, non_blocking=True), [Ap], [Ai], [Ax]
                else:
                    m_ = [m for i in range(batch_size)]
                    if pool:
                        def single_sample_fn_(args):
                            return single_sample_fn(*args)

                        outs = pool.starmap(single_sample_fn, zip(m_, Ax, Bflat, Ai, Aj))
                        out = [i for i in zip(*outs)]

                        out[0] = [i.to(device, dtype, non_blocking=True) for i in out[0]]
                        out[0] = torch.cat(out[0], dim=0)
                    else:
                        outs = [single_sample_fn(*i) for i in zip(m_, Ax, Bflat, Ai, Aj)]
                        out = [i for i in zip(*outs)]
                        out[0] = Bflat.to(device, dtype, non_blocking=True)
                return out
        else:
            def wrapper(Ap, Ai, Ax, Bflat):
                batch_size = Bflat.size(0)
                device = Bflat.device
                dtype = Bflat.dtype

                Bflat = Bflat.to('cpu',torch.float64, non_blocking=True)

                if batch_size == 1:
                    out = single_sample_fn(Ap[0], Ai[0], Ax[0], Bflat[0])
                    out = Bflat.to(device, dtype, non_blocking=True)
                else:
                    if pool:
                        def single_sample_fn_(args):
                            return single_sample_fn(*args)

                        outs = pool.starmap(single_sample_fn, zip(Ap, Ai, Ax, Bflat))
                        outs = [i.to(device, dtype, non_blocking=True) for i in outs]

                        out = torch.cat(outs, dim=0)
                    else:
                        [single_sample_fn(*i) for i in zip(Ap, Ai, Ax, Bflat)]
                        out = Bflat.to(device, dtype, non_blocking=True)
                return out

        return wrapper

    return decorator

def convert_matrix_and_solve_system(m, Ax, Bflat, Ai, Aj):
    Ap_Ai_Ax = _coo_to_csc(m, Ai, Aj, Ax)
    _klu_solve(Ap_Ai_Ax[0], Ap_Ai_Ax[1], Ap_Ai_Ax[2], Bflat)
    return Bflat, Ap_Ai_Ax[0], Ap_Ai_Ax[1], Ap_Ai_Ax[2]


def solve_system(Ap, Ai, Ax, Bflat):
    _klu_solve(Ap, Ai, Ax, Bflat)
    return Bflat

def LinearSolverSparse(num_workers=8, multiprocess=False, multithread=False):
    batch_process_A= batch_process(convert_matrix=True, num_workers=num_workers, multiprocess=multiprocess, multithread=multithread)
    batch_process_B = batch_process(convert_matrix=False, num_workers=num_workers, multiprocess=multiprocess, multithread=multithread)
    
    class LinearSolverSparse_(torch.autograd.Function):

        @staticmethod
        def forward(ctx, A, B):
            bs = B.size(0)
            m = B.size(1)
            n = B.size(2)

            Bflat = B.transpose(1, 2).reshape(bs, m * n)
            Ax = A._values().reshape(bs, -1)
            Ai = (A._indices()[1]).int().reshape(bs, -1)
            Aj = (A._indices()[2]).int().reshape(bs, -1)

            x, Ap, Ai, Ax = batch_process_A(convert_matrix_and_solve_system)(m, Ax, Bflat, Ai, Aj)
            x = x.view(bs, n, m).transpose(1, 2)
            ctx.save_for_backward(A, x, *Ap, *Ai, *Ax)
            return x

        @staticmethod
        def backward(ctx, grad):
            bs = grad.size(0)
            m = grad.size(1)
            n = grad.size(2)

            A = ctx.saved_tensors[0]
            x = ctx.saved_tensors[1]
            Ap = ctx.saved_tensors[2:2 + bs]
            Ai = ctx.saved_tensors[2 + bs:2 + 2 * bs]
            Ax = ctx.saved_tensors[2 + 2 * bs:2 + 3 * bs]

            A_ = A.transpose(1, 2)  # .coalesce()

            grad_flat = grad.transpose(1, 2).reshape(bs, m * n)

            # Do not use A transposed, but precomputed Ap, Ai, Ax already on the cpu, because A is symmetric
            # Ax = A_._values().reshape(bs, -1)
            # Ai = (A_._indices()[1]).int().reshape(bs, -1)
            # Aj = (A_._indices()[2]).int().reshape(bs, -1)

            gradB = batch_process_B(solve_system)(Ap, Ai, Ax, grad_flat).view(bs, n, m).transpose(1, 2)

            indices = A._indices()
            indices_ = A[0]._indices()
            gradA = torch.sparse_coo_tensor(indices, -torch.sum((gradB[:, indices_[0]] * x[:, indices_[1]]), -1).flatten(0, 1),
                                            (bs, m, m))

            return gradA, gradB

    return LinearSolverSparse_.apply


class SoftGraphCut(nn.Module):
    def __init__(self, img_size, num_workers=8, multiprocess=False, multithread=False):
        super().__init__()

        assert len(img_size) == 2
        self.size = img_size

        self.liner_system_solver = LinearSolverSparse(num_workers, multiprocess, multithread)

        self.pad_col_before = nn.ZeroPad2d((0, 0, 1, 0))
        self.pad_col_after = nn.ZeroPad2d((0, 0, 0, 1))
        self.pad_row_before = nn.ZeroPad2d((1, 0, 0, 0))
        self.pad_row_after = nn.ZeroPad2d((0, 1, 0, 0))

        self.indices = []
        # indices for K coefficients
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                target_index = j + i * img_size[1]
                self.indices.append([target_index, target_index])
        # indices for column coefficients
        for i in range(img_size[0] - 1):
            for j in range(img_size[1]):
                target_index = j + (i + 1) * img_size[1]
                coef_index = j + i * img_size[1]
                self.indices.append([target_index, coef_index])
        for i in range(img_size[0] - 1):
            for j in range(img_size[1]):
                target_index = j + i * img_size[1]
                coef_index = j + (i + 1) * img_size[1]
                self.indices.append([target_index, coef_index])
        # indices for row coefficients
        for i in range(img_size[0]):
            for j in range(img_size[1] - 1):
                target_index = j + 1 + i * img_size[1]
                coef_index = j + i * img_size[1]
                self.indices.append([target_index, coef_index])
        for i in range(img_size[0]):
            for j in range(img_size[1] - 1):
                target_index = j + i * img_size[1]
                coef_index = j + 1 + i * img_size[1]
                self.indices.append([target_index, coef_index])

    @lru_cache(maxsize=None)
    def indices_from_batch_size(self, bs):
        ret = [[[b] + i for i in self.indices] for b in range(bs)]
        ret = list(itertools.chain(*ret))
        return list(zip(*ret))

    def forward(self, pixel_scores, weights_col, weights_row):
        bs = pixel_scores.size(0)
        pixel_scores_size = tuple(pixel_scores.size())
        assert self.size == pixel_scores_size[-2:]
        assert len(weights_col.size()) == 3
        assert len(weights_row.size()) == 3
        assert bs == weights_col.size(0)
        assert bs == weights_row.size(0)
        assert self.size[0] == weights_col.size(1) + 1
        assert self.size[1] == weights_col.size(2)
        assert self.size[0] == weights_row.size(1)
        assert self.size[1] == weights_row.size(2) + 1

        B = torch.flatten(pixel_scores, -2, -1).swapaxes(1, -1)

        K = torch.ones_like(pixel_scores[:, 0, ...])

        weights_col_pad_before = self.pad_col_before(weights_col)
        weights_col_pad_after = self.pad_col_after(weights_col)
        weights_row_pad_before = self.pad_row_before(weights_row)
        weights_row_pad_after = self.pad_row_after(weights_row)

        K = K + weights_col_pad_before + weights_col_pad_after + weights_row_pad_before + weights_row_pad_after

        weights_col_coeff = -weights_col.flatten(-2, -1)
        weights_row_coeff = -weights_row.flatten(-2, -1)

        values = torch.cat((K.flatten(-2, -1), weights_col_coeff, weights_col_coeff, weights_row_coeff, weights_row_coeff), dim=1)
        values = values.flatten(0, 1)

        indices = self.indices_from_batch_size(bs)

        A = torch.sparse_coo_tensor(indices, values, (bs, self.size[0] * self.size[1], self.size[0] * self.size[1]))

        x = self.liner_system_solver(A, B).type_as(pixel_scores)
        return x.swapaxes(1, -1).view(pixel_scores_size)
