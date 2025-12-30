import torch
from typing import Optional


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits


def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean=True,
    loss=False,
):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results


def get_hit_at_10(
    logits: torch.Tensor,
    clean_logits: Optional[torch.Tensor],
    input_length: Optional[torch.Tensor],
    labels: torch.Tensor,
    mean=True,
    loss=False,
):
    assert (
        logits.dim() == 3
    ), "Logits should have shape (batch_size, seq_len, vocab_size)"
    assert logits.size(0) == labels.size(
        0
    ), "Batch size of logits and labels should match"
    assert logits.size(0) == 1, "Batch size should be 1 for hit@10 calculation"

    last_logit = logits[:, -1, :]
    _, topk_indices = torch.topk(last_logit, k=10, dim=-1)
    labels = labels.to(logits.device)
    hit = (topk_indices == labels[:, 0]).any()

    return hit
