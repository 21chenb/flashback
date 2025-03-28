import timeit
import os
import pandas as pd
from tqdm import tqdm
import jax
import numpy as np
from itertools import product
from pathlib import Path

from .correctness_utils import make_matrices, make_operators, Precision, \
    execute_operator

def get_walltime(b, r, c, d, out_d, num_heads, seed, std, bias, mqa, operator,
                 precision, order):
    assert mqa == False, "Not implemented yet"
    c = r
    out_d = d
    matrix_args = b, r, c, d, out_d, num_heads, seed, std, bias, mqa, precision
    Q, K, V, dot_dQ, dot_dK, dot_dV, dO = make_matrices(*matrix_args)
    test_args = Q, K, V, dot_dQ, dot_dK, dot_dV, dO

    def fn():
        return execute_operator(*test_args, operator, order)

    def blocked_fn():
        res = fn()
        jax.block_until_ready(res)

    # warmup
    for _ in range(2):
        blocked_fn()

    # time it
    return timeit.timeit(blocked_fn, number=5)

def main():
    # tests:
    def make_op_for(seq_length, d, precision, attn_type, naive):
        assert isinstance(precision, Precision)
        assert attn_type in ['softmax', 'sigmoid']
        sm_scale = d**(-0.5)
        causal = True
        attn_softcap = None
        sigmoid_bias = float(np.log(seq_length) * -2)
        operators = make_operators(sm_scale, attn_softcap, causal, precision,
                                     sigmoid_bias, attn_type)
        return operators[1] if naive else operators[0]

    # exps:
    # 1. Keep seq length at 512, vary d
    # 2. Keep d at 32, vary seq length from [32, 128, 256, 512, 1024, 2048, 4096]

    # ! EXP 1
    out_dired = Path('perf_output/')
    if not out_dired.exists():
        out_dired.mkdir()

    base_kw = dict(
        b=8, num_heads=16, seed=42,
        std=1, bias=0.1, mqa=False
    )

    EXP = os.environ['EXP']

    def exp1():
        df = []
        SL = 1024

        attn_types = ['softmax', 'sigmoid']
        precisions = [Precision.FP16]
        naives = [False, True]
        orders = [2, 1, 0]
        ds = [16, 32, 64, 128, 256]
        prod = list(product(attn_types, precisions, naives, orders, ds))
        for attn_type, precision, naive, order, d in tqdm(prod):
            kw = base_kw.copy()
            kw.update(dict(out_d=d, d=d, r=SL, c=SL))
            operator = make_op_for(SL, d, precision, attn_type, naive)
            walltime = get_walltime(operator=operator, precision=precision, order=order, **kw)
            df.append(
                dict(d=d, attn_type=attn_type, precision=precision.name,
                     walltime=walltime, order=order, naive=naive)
            )

        df = pd.DataFrame(df)
        df.to_csv(out_dired / 'exp1.csv')

    if EXP == '1':
        exp1()

    # ! EXP 2
    def exp2():
        df = []

        attn_types = ['softmax', 'sigmoid']
        precisions = [Precision.TF32_ROUND]
        naives = [False, True]
        orders = [2, 1, 0]
        sls = reversed([64, 128, 256, 512, 1024, 2048, 4096])
        prod = list(product(attn_types, precisions, naives, orders, sls))

        for attn_type, precision, naive, order, sl in tqdm(prod):
            kw = base_kw.copy()
            kw.update(dict(d=32, out_d=sl, r=sl, c=sl))
            operator = make_op_for(sl, kw['d'], precision, attn_type, naive)
            walltime = get_walltime(operator=operator, precision=precision, order=order, **kw)
            df.append(
                dict(r=sl, attn_type=attn_type, precision=precision.name,
                     walltime=walltime, order=order, naive=naive)
            )

        df = pd.DataFrame(df)
        df.to_csv(out_dired / 'exp2.csv')

    if EXP == '2':
        exp2()



if __name__ == '__main__':
    main()