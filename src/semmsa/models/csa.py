"""Cross-modal Spectral Alignment (CSA).

For each instance the L2-normalised representations V = [h_s, h_v, h_a, h_l] form an RBF kernel Gram matrix
K_ij = exp(-||h_i - h_j||^2 / (2 sigma^2)).  L_csa enhances the dominant eigenvalue of K through a softmax over
its spectrum, and L_sep separates the dominant semantic directions u_1 = V^T q_1 / ||V^T q_1|| of different
instances in a batch.
"""
import math

import torch
import torch.nn.functional as F


def kernel_gram(feats, sigma=1.0):
    """feats: (B, k, d) L2-normalised -> (B, k, k) RBF kernel Gram matrix."""
    diff = feats.unsqueeze(2) - feats.unsqueeze(1)
    sq = (diff * diff).sum(-1)
    return torch.exp(-sq / (2.0 * sigma ** 2))


def dominant_eigvec(K, iters=20):
    """Dominant eigenvector of a batch of PSD matrices with positive entries (differentiable power iteration)."""
    b, k, _ = K.shape
    q = torch.full((b, k, 1), 1.0 / math.sqrt(k), device=K.device, dtype=K.dtype)
    for _ in range(iters):
        q = torch.bmm(K, q)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return q.squeeze(-1)


def spectral_alignment_losses(feats, sigma=1.0, tau=0.1, power_iters=20):
    """feats: (B, k, d). Returns (L_csa, L_sep, eigenvalues (B, k) in ascending order)."""
    feats = F.normalize(feats.float(), dim=-1)
    K = kernel_gram(feats, sigma)
    lam = torch.linalg.eigvalsh(K)
    logits = lam / tau
    l_csa = -(logits[:, -1] - torch.logsumexp(logits, dim=-1)).mean()

    q1 = dominant_eigvec(K, power_iters)
    u1 = F.normalize(torch.einsum('bk,bkd->bd', q1, feats), dim=-1)
    sim = u1 @ u1.t()
    b = sim.size(0)
    if b > 1:
        off = sim - torch.diag_embed(torch.diagonal(sim))
        l_sep = (off * off).sum() / (b * (b - 1))
    else:
        l_sep = sim.new_zeros(())
    return l_csa, l_sep, lam
