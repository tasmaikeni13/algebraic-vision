"""Integer scalar-operation counts for fixed-shape JAX arithmetic programs.

Counts the actual traced forward/VJP program, including every fixed scan trip
and custom derivative body. It does not estimate time or count memory traffic.
Matrix products use the standard 2*M*N*K FLOP convention. No compiler-specific
CPU BF16 packing instructions are treated as mathematical floating-point work.
"""

from collections import Counter
from math import prod

import jax
import jax.numpy as jnp

CATEGORIES=("arithmetic","matmul","rsqrt","minmax","comparisons","selects","integer_arithmetic")
STRUCTURAL={"broadcast_in_dim","reshape","transpose","squeeze","concatenate","slice",
            "dynamic_slice","dynamic_update_slice","gather","pad","iota","convert_element_type",
            "copy","stop_gradient","device_put","rev","split","reduce_or","reduce_and",
            "and","or","not","xor","shift_left","shift_right_logical","shift_right_arithmetic",
            "bitcast_convert_type","eq_to_p","ad_checkpoint","optimization_barrier"}


def size(var):
    return prod(getattr(var.aval,"shape",()))


def floating(var):
    return jnp.issubdtype(getattr(var.aval,"dtype",jnp.int32),jnp.floating)


def count(closed):
    graph=closed.jaxpr if hasattr(closed,"jaxpr") else closed
    counts=Counter({name:0 for name in CATEGORIES})
    for equation in graph.eqns:
        name=equation.primitive.name
        out=equation.outvars
        n=sum(size(var) for var in out)
        if name=="scan":
            body=count(equation.params["jaxpr"])
            for key,val in body.items():counts[key]+=equation.params["length"]*val
        elif name in {"pjit","jit","custom_jvp_call","custom_vjp_call_jaxpr","closed_call","call","remat2"}:
            key=next((key for key in ["call_jaxpr","fun_jaxpr","jaxpr"] if key in equation.params),None)
            if key is None:
                raise ValueError(f"unrecognized call representation: {name}")
            counts.update(count(equation.params[key]))
        elif name=="dot_general":
            contracted=equation.params["dimension_numbers"][0][0]
            k=prod(equation.invars[0].aval.shape[i] for i in contracted)
            counts["matmul"]+=2*k*n
        elif name=="reduce_sum":
            counts["arithmetic"]+=size(equation.invars[0])-n
        elif name in {"reduce_max","reduce_min"}:
            counts["minmax"]+=size(equation.invars[0])-n
        elif name in {"add","add_any","sub","mul","div","neg","abs","rem"}:
            counts["arithmetic" if floating(out[0]) else "integer_arithmetic"]+=n
        elif name=="integer_pow":
            power=equation.params["y"]
            magnitude=abs(power)
            multiplications=max(0,magnitude.bit_length()+magnitude.bit_count()-2)
            counts["arithmetic"]+=n*(multiplications+(power<0))
        elif name=="rsqrt":
            counts["rsqrt"]+=n
        elif name in {"max","min"}:
            counts["minmax"]+=n
        elif name in {"lt","le","gt","ge","eq","ne"}:
            counts["comparisons"]+=n
        elif name=="sign":
            # sign is the arithmetic alias select(x>0,1,select(x<0,-1,0)).
            counts["comparisons"]+=2*n
            counts["selects"]+=2*n
        elif name=="select_n":
            counts["selects"]+=n
        elif name=="scatter-add":
            counts["arithmetic"]+=size(equation.invars[2])
        elif name=="scatter":
            # Pure replacement has no arithmetic; other updates must be explicit.
            updater=equation.params.get("update_jaxpr")
            if updater is not None and getattr(updater,"eqns",[]):
                raise ValueError("arithmetic scatter update requires a supported counter")
        elif name in STRUCTURAL:
            pass
        else:
            raise ValueError(f"unsupported primitive {name}; refusing an incomplete count")
    return dict(counts)


def forward_backward(fn,arguments,cotangent):
    def both(*args):
        result,pull=jax.vjp(fn,*args[:-1])
        return result,pull(args[-1])
    return count(jax.make_jaxpr(both)(*arguments,cotangent))
