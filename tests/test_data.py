"""Input pipeline: data order and evaluation coverage."""

import numpy as np

from algebraic_vision import data


def test_sampler_is_topology_independent():
    sampler = data.EpochSampler(np.arange(1000), seed=7)
    whole = sampler.positions(3 * 64, 64)
    parts = [data.EpochSampler(np.arange(1000), 7).positions(3 * 64 + 16 * i,
                                                             16)
             for i in range(4)]
    np.testing.assert_array_equal(whole, np.concatenate(parts))
    epoch0 = sampler.positions(0, 1000)
    assert sorted(epoch0.tolist()) == list(range(1000))
    assert not np.array_equal(epoch0, sampler.positions(1000, 1000))


def test_eval_batches_cover_each_index_once():
    class Source:
        def load(self, idx):
            return np.asarray(idx)[:, None], np.asarray(idx)

    seen = []
    for _, labels, valid in data.eval_batches(Source(), np.arange(23), 8):
        seen.extend(labels[valid].tolist())
    assert seen == list(range(23))
