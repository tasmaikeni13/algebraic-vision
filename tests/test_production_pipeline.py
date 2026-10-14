"""Exercise paired augmentations, distributed update, resume and reports."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from algebraic_vision.production import ProductionConfig, build_steps, config_hash, host_schedule, optimizer_init, restore_checkpoint, save_checkpoint, target_loss
from algebraic_vision.regularization import prepare_batch
from algebraic_vision.train import build_model_config
from algebraic_vision.model import init_params
from experiments.production import jobs


def test_regularization_replays_across_arms_and_mixed_targets_are_distributions():
    x = jnp.asarray(np.random.default_rng(3).integers(0,256,(4,24,24,3),dtype=np.uint8))
    labels = jnp.arange(4)
    key = jax.random.PRNGKey(17)
    results = []
    for arm in ("standard", "algebraic", "algebraic_v2"):
        cfg = ProductionConfig(arm=arm, image_size=16, randaugment_layers=2, mixup_alpha=0.8, cutmix_alpha=1)
        result = jax.jit(lambda x,y,k:prepare_batch(cfg,7,x,y,k,jnp.float32(0.6)))(x,labels,key)
        results.append(result)
        assert result[0].shape == (4,16,16,3)
        np.testing.assert_allclose(jnp.sum(result[1],-1),1,atol=1e-6)
        assert bool(jnp.all(jnp.isfinite(result[0])))
    for first, other in zip(results[0], results[2]):
        np.testing.assert_array_equal(first,other)


@pytest.mark.parametrize("arm", ["standard", "algebraic", "algebraic_v2"])
def test_update_checkpoint_resume_and_full_graph(arm,tmp_path):
    cfg = ProductionConfig(arm=arm,total_steps=3,warmup_steps=0,image_size=16,batch_size=2,eval_batch=2,
              require_pod=False,loss_backend="xla", stochastic_depth=0.1,
              model=dict(width=32,depth=2,heads=1,mlp_dim=64,patch_size=8,num_classes=7,compute_dtype="float32",attention_backend="xla"))
    mc = build_model_config(cfg)
    mesh = Mesh(np.asarray(jax.devices()[:1]),("data",))
    rep, shard = NamedSharding(mesh,P()),NamedSharding(mesh,P("data"))
    params = jax.device_put(init_params(mc,jax.random.PRNGKey(42)),rep)
    state = jax.device_put(optimizer_init(params),rep)
    update, evaluate, _ = build_steps(cfg,mc,mesh,donate=False)
    x = jax.device_put(jnp.ones((2,16,16,3),jnp.float32)*0.25,shard)
    labels = jax.device_put(jnp.array([1,2]),shard)
    targets = jax.device_put(jax.nn.one_hot(labels,7)*0.9+0.1/7,shard)
    key = jax.random.PRNGKey(31)
    scalars = host_schedule(cfg,0)
    from scripts.audit_production import scan_hlo,scan_jaxpr
    args = (params,state,x,targets,labels,scalars,key)
    compiled = update.lower(*args).compile()
    assert scan_hlo(compiled.as_text())["pure"] == (arm != "standard")
    if arm != "standard":
        assert scan_jaxpr(jax.make_jaxpr(update)(*args))["pure"]
    first, firststate, metrics = update(*args)
    path = tmp_path/"checkpoint.pkl"
    save_checkpoint(path,cfg,first,firststate,1,[{"step":1}])
    saved = restore_checkpoint(path,cfg)
    assert saved["step"] == 1
    uninterrupted = update(first,firststate,x,targets,labels,host_schedule(cfg,1),key)
    resumed = update(jax.device_put(saved["params"],rep),jax.device_put(saved["opt_state"],rep),x,targets,labels,host_schedule(cfg,1),key)
    for a,b in zip(jax.tree.leaves(uninterrupted),jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a,b)
    res = evaluate(resumed[0],jnp.full((2,16,16,3),127,jnp.uint8),labels,jnp.array([True,False]))
    assert float(res["count"]) == 1
    assert bool(jnp.isfinite(res["nll"]))
    cfg.seed += 1
    with pytest.raises(ValueError,match="differs"):
        restore_checkpoint(path,cfg)


def test_report_and_selection_require_complete_paired_evidence(tmp_path):
    from analysis.production_report import report
    from analysis.production_select import select
    matrix = jobs("smoke")
    records = []
    for c in matrix:
        arm = ["standard","algebraic","algebraic_v2"].index(c["arm"])
        metrics = dict(top1=0.5+0.01*arm+0.001*(c["seed"]-42),nll=2-arm*0.1,ece=0.05-arm*0.001)
        records.append({"config":c,"complete":True,"nonfinite_steps":0,"devices":16,"processes":4,
                        "source_sha256":"test fixture","dataset_metadata":{"synthetic":True},
                        "results":{"minival":metrics,"val":metrics},"diagnostics":[{"attention_entropy":1.,"mean_example_median_abs_score":0.8}],
                        "train_images_per_sec":1000})
    result = report(matrix,records,tmp_path)
    assert result["synthetic"]
    assert result["scales"]["vit-b16"]["comparisons"]["algebraic_v2_minus_standard"]["top1"]["mean_difference"] > 0
    assert (tmp_path/"vit-l16_metrics.pdf").exists()
    with pytest.raises(ValueError,match="entire"):
        report(matrix,records[:-1],tmp_path)
    with pytest.raises(ValueError,match="held-out"):
        select(matrix,records)
    selected=select(matrix,records,allow_synthetic=True)
    assert selected["synthetic"] and len(selected["choices"])==2
