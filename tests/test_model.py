"""The network: output shape and size, indifference to band order and to padding, older
networks reading the fields they know, folding FiLM into the GroupNorm without changing the
output, and gradients reaching every part."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catena.descriptor import N_CARD, N_FIELDS  # noqa: E402
from catena.model import Clarity, FiLM, count_parameters, fold_block  # noqa: E402


def small(train: bool = False):
    torch.manual_seed(0)
    net = Clarity(widths=(8, 16, 16, 16), stem=8)
    return net.train() if train else net.eval()


def inputs(batch=2, bands=5, size=24):
    g = torch.Generator().manual_seed(1)
    x = torch.rand(batch, bands, size, size, generator=g)
    d = torch.randn(batch, bands, N_FIELDS, generator=g)
    real = torch.ones(batch, bands, dtype=torch.bool)
    return x, d, real


def test_shape_and_size():
    net = Clarity().eval()
    assert count_parameters(net) == 2_185_141  # 4.4 MB at float16; the budget is 6 MB
    x, d, real = inputs(size=48)
    real[1, 3:] = False  # the second image has 3 bands
    with torch.no_grad():
        assert net(x, d, real).shape == (2, 4, 48, 48)


def test_the_model_that_reads_the_card():
    net = Clarity(fields=N_CARD).eval()
    assert count_parameters(net) == 2_184_565  # 6 numbers per band instead of 15
    x, d, real = inputs(size=24)
    with torch.no_grad():
        assert net(x, d[..., :N_CARD], real).shape == (2, 4, 24, 24)


def test_band_order_and_padding_change_nothing():
    net = small()
    x, d, real = inputs()
    with torch.no_grad():
        out = net(x, d, real)
        perm = torch.tensor([3, 0, 4, 1, 2])
        assert torch.allclose(net(x[:, perm], d[:, perm], real[:, perm]), out, atol=1e-5)
        junk = torch.cat([x, 7 * torch.rand(2, 3, 24, 24)], dim=1)  # 3 empty slots
        djunk = torch.cat([d, torch.randn(2, 3, N_FIELDS)], dim=1)
        rjunk = torch.cat([real, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
        assert torch.allclose(net(junk, djunk, rjunk), out, atol=1e-5)


def test_a_network_with_fewer_fields_reads_the_first_ones():
    torch.manual_seed(0)
    old = Clarity(widths=(8, 16, 16, 16), stem=8, fields=N_FIELDS - 1).eval()  # the pilot's
    assert old.bands.embed[0].weight.shape[1] == N_FIELDS - 1
    x, d, real = inputs()
    with torch.no_grad():
        assert torch.equal(old(x, d, real), old(x, d[..., :-1], real))


def test_folding_keeps_the_output():
    net = small()
    for m in net.modules():  # give every FiLM something to do
        if isinstance(m, FiLM):
            torch.nn.init.normal_(m.to_gb.weight, std=0.1)
            torch.nn.init.normal_(m.to_gb.bias, std=0.1)
    x, d, real = inputs(batch=1)
    with torch.no_grad():
        before = net(x, d, real)
        _, z = net.bands(x, d, real)
        for level in [*net.down, *net.up]:
            for block in level:
                fold_block(block, z)
        after = net(x, d, real)
    assert torch.allclose(before, after, atol=1e-5)


def test_gradients_reach_every_part():
    net = small(train=True)
    x, d, real = inputs()
    net(x, d, real).square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
