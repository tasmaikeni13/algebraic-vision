"""Check that the pod forms one 16-device mesh and that a psum works.

Run on every host at once: scripts/pod_run.sh LOG .venv/bin/python
scripts/pod_check.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.distributed.initialize()
mesh = Mesh(np.asarray(jax.devices()), ("data",))
x = jax.make_array_from_process_local_data(
    NamedSharding(mesh, P("data")),
    np.full((jax.local_device_count(),), jax.process_index() + 1.0,
            np.float32), (jax.device_count(),))
total = jax.jit(jnp.sum, out_shardings=NamedSharding(mesh, P()))(x)
expected = 4 * sum(range(1, jax.process_count() + 1))
print(f"process {jax.process_index()}/{jax.process_count()} local "
      f"{jax.local_device_count()} global {jax.device_count()} "
      f"sum {float(total)} (expected {expected})", flush=True)
