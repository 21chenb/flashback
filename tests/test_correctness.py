import pytest
import itertools
from .correctness_utils import make_operators, one_test
import jax
from functools import partial, cache
from flashback.pallas_utils import Precision
import numpy as np
from flashback import autotune

# check if testing
COUNT = 0
VARNAMES = "b,r,c,d,out_d,num_heads,seed,std,bias,attn_type,print_debug,test,order,sm_scale,causal,attn_softcap,precision,sigmoid_bias"

INPUTS = {
    "b": [16],
    "r": [32, 128, 256],
    "c": [None],
    "d": [32],
    "out_d": [None],
    "num_heads": [8, 16],
    "seed": [42],
    "std": [0.01],
    "bias": [0.1],
    "print_debug": [False],
    "test": [True],
    "order": [2, 1, 0],
    "sm_scale": [32**(-0.5)],
    "causal": [True],
    "attn_softcap": [None],
    "precision": [Precision.FP32], # , 'fp16', 'tf32-round'], #'fp16', 'fp32', 'tf32-truncate', 'tf32-round'],
    "attn_type": ['softmax', 'sigmoid'],
}

def make_product():
    # add last slot for sigmoid_bias
    one_product = [params + (float(np.log(params[1]) * -2),) for params in itertools.product(
        INPUTS["b"],
        INPUTS["r"],
        INPUTS["c"],
        INPUTS["d"],
        INPUTS["out_d"],
        INPUTS["num_heads"],
        INPUTS["seed"],
        INPUTS["std"],
        INPUTS["bias"],
        INPUTS["attn_type"],
        INPUTS["print_debug"],
        INPUTS["test"],
        INPUTS["order"],
        INPUTS["sm_scale"],
        INPUTS["causal"],
        INPUTS["attn_softcap"],
        INPUTS["precision"]
    )]


    assert len(one_product[0]) == 18
    return one_product


@cache
def combine_with_precision(f, prec, causal):
    return partial(f, precision=prec, causal=causal)

def make_ids(*args, **kwargs):
    global COUNT
    keys = VARNAMES.split(",")
    idx = COUNT % len(keys)
    COUNT += 1
    name = keys[idx]
    value = args[0]
    return f'{name}={value}'

@pytest.mark.parametrize(VARNAMES, make_product(), ids=make_ids)
def test_main(b, r, c, d, out_d, num_heads, seed, std, bias, attn_type, print_debug,
              test, order, sm_scale, causal, attn_softcap, precision, sigmoid_bias):
    autotune.SKIP_AUTOTUNE = True
    operator, gt_operator = make_operators(sm_scale, attn_softcap, causal,
                                           precision, sigmoid_bias, attn_type)
    # one_test(b, r, c, d, out_d, num_heads, seed, std, bias, print_debug, test,
    #          order, operator, gt_operator, precision) 
    one_test(**dict(
        b=b, r=r, c=c, d=d, out_d=out_d, num_heads=num_heads, seed=seed, std=std,
        bias=bias, mqa=False, print_debug=print_debug, test=test, order=order,
        operator=operator, gt_operator=gt_operator, precision=precision
    ))
    autotune.SKIP_AUTOTUNE = False
