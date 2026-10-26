import torch

from semmsa.models.adapter import PromptAdapter
from semmsa.models.layers import Transformer


def test_transformer_prepends_summary_tokens():
    enc = Transformer(num_frames=20, token_len=8, dim=32, depth=2, heads=4, mlp_dim=64)
    x = torch.randn(3, 20, 32)
    mask = torch.zeros(3, 20, dtype=torch.bool)
    mask[:, 10:] = True
    out = enc(x, key_padding_mask=mask)
    assert out.shape == (3, 28, 32)
    assert torch.isfinite(out).all()


def test_masked_frames_do_not_change_summary_tokens():
    torch.manual_seed(0)
    enc = Transformer(num_frames=20, token_len=8, dim=32, depth=2, heads=4, mlp_dim=64).eval()
    x = torch.randn(2, 20, 32)
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, 12:] = True
    y = x.clone()
    y[:, 12:] = torch.randn(2, 8, 32)
    with torch.no_grad():
        assert torch.allclose(enc(x, mask)[:, :8], enc(y, mask)[:, :8], atol=1e-5)


def test_prompt_adapter_output_shape():
    adapter = PromptAdapter(in_dim=20, dim=64, out_dim=48, num_prompts=8, depth=2, heads=4)
    out = adapter(torch.randn(4, 50, 20))
    assert out.shape == (4, 8, 48)
    out.sum().backward()
    assert adapter.prompts.grad is not None
