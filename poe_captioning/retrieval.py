from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


TEMPERATURE = 0.01


def sibling_lists(train_idx: torch.Tensor, source_ids: torch.Tensor) -> list[list[int]]:
    src = source_ids.cpu().long()
    rows = [int(x) for x in train_idx.cpu().long().tolist()]
    by_src: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_src[int(src[row])].append(row)
    out = []
    for row in rows:
        sibs = [j for j in by_src[int(src[row])] if j != row]
        out.append(sibs)
    return out


def build_sibling_queries(
    embs: torch.Tensor,
    train_idx: torch.Tensor,
    source_ids: torch.Tensor,
    n_sib: int | None,
    dirichlet: bool,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    rng = np.random.default_rng(int(seed))
    rows = [int(x) for x in train_idx.cpu().long().tolist()]
    groups = sibling_lists(train_idx, source_ids)
    queries = torch.empty(len(rows), embs.shape[-1], dtype=torch.float32)
    n_fallback = 0
    n_used = []
    for i, row in enumerate(rows):
        sibs = list(groups[i])
        if n_sib is not None and len(sibs) > int(n_sib):
            pick = rng.choice(len(sibs), size=int(n_sib), replace=False)
            sibs = [sibs[int(p)] for p in pick]
        if not sibs:
            queries[i] = embs[row].float()
            n_fallback += 1
            n_used.append(0)
            continue
        vecs = embs[torch.tensor(sibs, dtype=torch.long)].float()
        if dirichlet:
            w = torch.from_numpy(rng.dirichlet(np.ones(len(sibs))).astype("float32"))
        else:
            w = torch.full((len(sibs),), 1.0 / len(sibs), dtype=torch.float32)
        queries[i] = (w.unsqueeze(-1) * vecs).sum(dim=0)
        n_used.append(len(sibs))
    meta = {
        "n_query": len(rows),
        "n_fallback_self": n_fallback,
        "n_sib": n_sib,
        "dirichlet": bool(dirichlet),
        "mean_n_used": float(np.mean(n_used) if n_used else 0.0),
        "seed": int(seed),
    }
    return queries, meta


def _as_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.to(dtype=dtype)
    if device is not None:
        x = x.to(device)
    return x


def _forbidden_logits(
    logits: torch.Tensor,
    src: torch.Tensor,
    query_src: torch.Tensor,
    forbidden: torch.Tensor | None,
) -> torch.Tensor:
    same = query_src.unsqueeze(1).eq(src.unsqueeze(0))
    logits = logits.masked_fill(same, float("-inf"))
    if forbidden is not None:
        logits = logits.masked_fill(forbidden.unsqueeze(0), float("-inf"))
    return logits


def precompute_query_neighbors(
    queries: torch.Tensor | np.ndarray,
    embs: torch.Tensor | np.ndarray,
    source_ids: np.ndarray | torch.Tensor,
    query_source_ids: np.ndarray | torch.Tensor,
    k: int,
    temperature: float = TEMPERATURE,
    chunk: int = 256,
    device: str = "cuda",
    forbidden: np.ndarray | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    mem = _as_tensor(embs, device=device)
    src = _as_tensor(source_ids, device=device, dtype=torch.int64)
    q_all = _as_tensor(queries, device=device)
    q_src = _as_tensor(query_source_ids, device=device, dtype=torch.int64)
    n = mem.shape[0]
    forbid = None
    if forbidden is not None:
        forbid = _as_tensor(forbidden, device=device, dtype=torch.bool)
        if forbid.numel() != n:
            raise ValueError("forbidden mask must match memory rows")
    mem_n = F.normalize(mem.float(), dim=-1)
    q_n = F.normalize(q_all.float(), dim=-1)
    all_index, all_pi, all_mass = [], [], []
    nq = int(q_n.shape[0])
    k = min(int(k), max(n - 1, 1))
    for start in range(0, nq, chunk):
        end = min(start + chunk, nq)
        logits = q_n[start:end] @ mem_n.T
        logits = _forbidden_logits(logits, src, q_src[start:end], forbid)
        full_pi = torch.softmax(logits / temperature, dim=-1)
        weights, index = full_pi.topk(k, dim=-1, sorted=True)
        mass = weights.sum(dim=-1)
        pi = weights / mass.clamp_min(1e-12).unsqueeze(-1)
        all_index.append(index.cpu())
        all_pi.append(pi.cpu())
        all_mass.append(mass.cpu())
        del logits, full_pi
    return {
        "index": torch.cat(all_index, dim=0),
        "pi": torch.cat(all_pi, dim=0),
        "mass": torch.cat(all_mass, dim=0),
        "k": torch.tensor(k),
        "temperature": torch.tensor(temperature),
    }


def precompute_query_mix(
    queries: torch.Tensor | np.ndarray,
    embs: torch.Tensor | np.ndarray,
    source_ids: np.ndarray | torch.Tensor,
    query_source_ids: np.ndarray | torch.Tensor,
    temperature: float = TEMPERATURE,
    chunk: int = 128,
    device: str = "cuda",
    forbidden: np.ndarray | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    mem = _as_tensor(embs, device=device)
    src = _as_tensor(source_ids, device=device, dtype=torch.int64)
    q_all = _as_tensor(queries, device=device)
    q_src = _as_tensor(query_source_ids, device=device, dtype=torch.int64)
    n = mem.shape[0]
    forbid = None
    if forbidden is not None:
        forbid = _as_tensor(forbidden, device=device, dtype=torch.bool)
        if forbid.numel() != n:
            raise ValueError("forbidden mask must match memory rows")
    mem_n = F.normalize(mem.float(), dim=-1)
    q_n = F.normalize(q_all.float(), dim=-1)
    prefixes, masses = [], []
    nq = int(q_n.shape[0])
    for start in range(0, nq, chunk):
        end = min(start + chunk, nq)
        logits = q_n[start:end] @ mem_n.T
        logits = _forbidden_logits(logits, src, q_src[start:end], forbid)
        pi = torch.softmax(logits / temperature, dim=-1)
        mixed = pi @ mem.float()
        prefixes.append(F.normalize(mixed, dim=-1).cpu())
        masses.append(torch.ones(end - start, dtype=torch.float32))
        del logits, pi, mixed
    prefix = torch.cat(prefixes, dim=0)
    return {
        "prefix": prefix,
        "mass": torch.cat(masses, dim=0),
        "temperature": torch.tensor(temperature),
    }


def embed_mean(embs_raw: torch.Tensor, pi: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    mixed = (pi.unsqueeze(-1) * embs_raw.float()).sum(dim=-2)
    if normalize:
        mixed = F.normalize(mixed, dim=-1)
    return mixed
