import torch

from mmface.models import LatentDiscriminator, MultimodalGenerator
from mmface.models.blocks import MinibatchStdDev, TextCrossAttention


def test_shapes():
    G = MultimodalGenerator(base_channels=16, max_channels=64, levels=4)
    D = LatentDiscriminator(base_channels=16, max_channels=64, levels=4)
    sketch = torch.randn(2, 4, 32, 32)
    out = G(sketch, torch.randn(2, 512))
    assert out.shape == sketch.shape
    assert D(out, sketch).shape == (2,)


def test_ablation_switches():
    G = MultimodalGenerator(
        base_channels=8, max_channels=32, levels=3,
        use_adagn=False, use_skip_connections=False,
        noise_injection=False, use_mapping_network=False,
    )
    assert G(torch.randn(1, 4, 32, 32), torch.randn(1, 512)).shape == (1, 4, 32, 32)


def test_cross_attention():
    G = MultimodalGenerator(base_channels=8, max_channels=32, levels=3, use_cross_attention=True)
    tokens = torch.randn(1, 77, 512)
    mask = torch.zeros(1, 77, dtype=torch.long)
    mask[:, :10] = 1
    out = G(torch.randn(1, 4, 32, 32), torch.randn(1, 512), tokens, mask)
    assert out.shape == (1, 4, 32, 32)


def test_attention_maps_are_kept():
    attn = TextCrossAttention(32)
    attn.keep_attention = True
    attn(torch.randn(2, 32, 8, 8), torch.randn(2, 77, 512))
    w = attn.last_attention
    assert w.shape == (2, 8, 8, 77)
    assert torch.allclose(w.sum(-1), torch.ones(2, 8, 8), atol=1e-5)


def test_minibatch_std_adds_channel():
    assert MinibatchStdDev(4)(torch.randn(8, 16, 4, 4)).shape == (8, 17, 4, 4)
