from flashback.pallas_utils import Precision
import jax.numpy as jnp
import jax
from functools import partial
import pytest
from flashback.ops import sigmoid_mha, softmax_mha
from flashback.ops import naive_sigmoid_mha, naive_softmax_mha

def get_tols(precision):
    if precision == Precision.FP16 or precision == Precision.TF32_ROUND:
        rtol = 1e-3
    elif precision == Precision.TF32_TRUNC:
        rtol = 1e-2
    elif precision == Precision.FP32:
        rtol = 1e-4
    else:
        raise ValueError(f"Unknown precision: {precision}")

    rtol *= 10
    return rtol, 1e-1

@partial(jax.jit, static_argnames=('order', 'operator'))
def execute_operator(Q, K, V, dot_dQ, dot_dK, dot_dV, dO, operator, order):
    def fwd_pass(Q, K, V):
        output = operator(Q, K, V)
        return jnp.log((output * dO).mean()**2)

    def get_grads(Q, K, V):
        dQ, dK, dV = jax.grad(fwd_pass, argnums=[0, 1, 2])(Q, K, V)
        return (dQ * dot_dQ).mean() + (dK * dot_dK).mean() + (dV * dot_dV).mean()

    if order == 2:
        dot_dQ, dot_dK, dot_dV = jax.grad(get_grads, argnums=[0, 1, 2])(Q, K, V) # f_vjp(1.0)
        return dot_dQ, dot_dK, dot_dV
    elif order == 1:
        dQ, dK, dV = jax.grad(fwd_pass, argnums=[0, 1, 2])(Q, K, V)
        return dQ, dK, dV
    elif order == 0:
        return operator(Q, K, V)

def make_matrices(b, r, c, d, out_d, num_heads, seed, std, bias, mqa, precision):
    def make_matrix(shape, seed):
        mat = jax.random.exponential(jax.random.PRNGKey(seed), shape, dtype=jnp.float32)
        ret = (mat - 1) * std + bias
        if precision == Precision.FP16:
            dtype = jnp.float16
        elif precision == Precision.BF16:
            dtype = jnp.bfloat16
        else:
            dtype = jnp.float32

        return ret.astype(dtype)

    def make_matrix_like(mat, seed):
        return make_matrix(mat.shape, seed)

    Q = make_matrix((b, r, num_heads, d), seed)
    if not mqa:
        K = make_matrix((b, c, num_heads, d), seed+1)
        V = make_matrix((b, c, num_heads, out_d), seed+2)
    else:
        K = make_matrix((b, c, d), seed+1)
        V = make_matrix((b, c, out_d), seed+2)

    dot_dQ = make_matrix_like(Q, seed+4)
    dot_dK = make_matrix_like(K, seed+5)
    dot_dV = make_matrix_like(V, seed+6)

    dO = make_matrix((b, r, num_heads, out_d), seed+3)
    return Q, K, V, dot_dQ, dot_dK, dot_dV, dO


def test_operator_match(Q, K, V, dot_dQ, dot_dK, dot_dV, dO, print_debug, test,
                        order, operator, gt_operator, precision):
    # Run both operators
    fa = execute_operator(Q, K, V, dot_dQ, dot_dK, dot_dV, dO, operator, order)
    gt = execute_operator(Q, K, V, dot_dQ, dot_dK, dot_dV, dO, gt_operator, order)

    # want lower precision for higher order
    rtol, atol = get_tols(precision)
    rtol = rtol / 10**(2 - order)
    atol = atol / 10**(2 - order)

    # Check if the average error is within the tolerance
    def check_tensor(t_fa, t_gt, name):
        diff = jnp.abs(t_fa - t_gt)
        rdiff = jnp.mean(diff/(jnp.abs(t_gt).mean() + 1e-5))
        adiff = jnp.mean(diff)

        if print_debug:
            msg = f'>> {name} '
            jax.debug.print(msg + 'relative diff: {rdiff} vs {rtol}', rdiff=rdiff,
                            rtol=rtol, ordered=True)
            jax.debug.print(msg + 'absolute diff: {adiff} vs {atol}', adiff=adiff,
                            atol=atol, ordered=True)

        if test:
            rdiff_msg = f">> {name}: Bad relative diff: {rdiff} vs {rtol}; abs diff: {adiff} vs {atol}"
            # try:
            assert rdiff <= rtol, rdiff_msg
            # except:
            #     print(rdiff_msg)
                # import pdb; pdb.set_trace()

    if order == 2:
        check_tensor(fa[0], gt[0], 'dot_dQ')
        check_tensor(fa[1], gt[1], 'dot_dK')
        check_tensor(fa[2], gt[2], 'dot_dV')
    elif order == 1:
        check_tensor(fa[0], gt[0], 'dQ')
        check_tensor(fa[1], gt[1], 'dK')
        check_tensor(fa[2], gt[2], 'dV')
    elif order == 0:
        check_tensor(fa, gt, 'O')

def make_operators(sm_scale, attn_softcap, causal, precision, sigmoid_bias,
                   attn_type):
    kw = {
        'sm_scale': sm_scale,
        'causal': causal,
        'precision': precision,
        'bias': sigmoid_bias
    }

    if attn_type == 'softmax':
        del kw['bias']
        operator, gt_operator = softmax_mha, naive_softmax_mha
    else:
        operator, gt_operator = sigmoid_mha, naive_sigmoid_mha

    operator = partial(operator, **kw)
    gt_operator = partial(gt_operator, **kw)
    return operator, gt_operator

def one_test(b, r, c, d, out_d, num_heads, seed, std, bias, mqa,
             print_debug, test, order, operator, gt_operator, precision,
             test_op=test_operator_match):
    assert mqa == False, "Not implemented yet"
    c = r
    out_d = d
    matrix_args = b, r, c, d, out_d, num_heads, seed, std, bias, mqa, precision
    Q, K, V, dot_dQ, dot_dK, dot_dV, dO = make_matrices(*matrix_args)

    test_args = Q, K, V, dot_dQ, dot_dK, dot_dV, dO
    return test_op(*test_args, print_debug, test, order, operator, gt_operator,
                   precision)

if __name__ == '__main__':
    op, gt_op = make_operators(None, None, True, Precision.FP32, 0.0, 'sigmoid')
    one_test(8, 32, 32, 32, 32, 4, 0, 1.0, 0.0, False, True, False, 2, op, gt_op,
             precision=Precision.FP32)

# def test_main(b, r, c, d, out_d, num_heads, seed, std, bias, mqa, print_debug,
#               test, order, sm_scale, causal, attn_softcap, precision):
#     """
#     Test the main function with the Cartesian product of the specified input sets.
#     """
    

#     operator = make_operator(sm_scale, attn_softcap, causal, precision)
#     operator = jax.tree_util.Partial(operator)
#     gt_operator = combine_with_precision(attn_forward, precision, causal)
#     gt_operator = jax.tree_util.Partial(gt_operator, sm_scale=sm_scale,
#                                         attn_softcap=attn_softcap)

#     full_test(b, r, c, d, out_d, num_heads, seed, std, bias, mqa, print_debug,
#               test, order, operator, gt_operator, precision) 

