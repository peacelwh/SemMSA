import torch
import torch.nn.functional as F

from semmsa.models.csa import dominant_eigvec, kernel_gram, spectral_alignment_losses


def test_kernel_gram_is_symmetric_with_unit_diagonal():
    feats = F.normalize(torch.randn(3, 4, 16), dim=-1)
    K = kernel_gram(feats, sigma=1.0)
    assert K.shape == (3, 4, 4)
    assert torch.allclose(K, K.transpose(1, 2), atol=1e-6)
    assert torch.allclose(torch.diagonal(K, dim1=1, dim2=2), torch.ones(3, 4), atol=1e-6)


def test_power_iteration_matches_eigh():
    feats = F.normalize(torch.randn(5, 4, 32, dtype=torch.float64), dim=-1)
    K = kernel_gram(feats)
    q = dominant_eigvec(K, iters=100)
    ref = torch.linalg.eigh(K).eigenvectors[..., -1]
    assert torch.allclose(q.abs(), ref.abs(), atol=1e-6)


def test_losses_are_finite_and_differentiable():
    feats = torch.randn(8, 4, 32, requires_grad=True)
    l_csa, l_sep, lam = spectral_alignment_losses(feats, sigma=1.0, tau=0.1)
    assert lam.shape == (8, 4)
    assert torch.isfinite(l_csa) and torch.isfinite(l_sep)
    (l_csa + l_sep).backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()


def test_aligned_representations_lower_csa():
    base = torch.randn(8, 1, 32)
    aligned = base.repeat(1, 4, 1) + 0.01 * torch.randn(8, 4, 32)
    scattered = torch.randn(8, 4, 32)
    assert spectral_alignment_losses(aligned)[0] < spectral_alignment_losses(scattered)[0]


def test_single_instance_has_zero_separation_loss():
    _, l_sep, _ = spectral_alignment_losses(torch.randn(1, 4, 32))
    assert l_sep.item() == 0.0
