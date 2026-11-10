import argparse
import random
from collections import Counter
from pathlib import Path


def train(args):
    import clip
    import numpy as np
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm
    from transformers import get_linear_schedule_with_warmup

    from .backend import MAX_LEN, load_decoder, sequence_logprob
    from .data import atomic_output, coco_memory_groups, flickr_train_memory, load_h5_memory
    from .retrieval import build_sibling_queries, embed_mean, precompute_query_mix, precompute_query_neighbors

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.corpus == "coco":
        raw_embeddings, texts = load_h5_memory(args.memory)
        train_rows, group_ids = coco_memory_groups(texts, args.captions, args.text_groups)
        embeddings = torch.from_numpy(np.ascontiguousarray(raw_embeddings))
    else:
        embeddings, texts, group_ids = flickr_train_memory(args.memory, args.captions, args.model, device)
        train_rows = np.arange(len(texts), dtype=np.int64)

    if embeddings.shape != (len(texts), 768):
        raise ValueError("Memory rows must contain one 768-dimensional embedding per caption.")
    if len(train_rows) < args.batch_size:
        raise ValueError("The training split has fewer captions than the batch size.")
    if len(set(group_ids[train_rows])) < 2:
        raise ValueError("Training requires at least two eligible source groups.")

    train_idx = torch.as_tensor(train_rows, dtype=torch.long)
    source_ids = torch.as_tensor(group_ids, dtype=torch.long)
    forbidden = torch.ones(len(texts), dtype=torch.bool)
    forbidden[train_idx] = False
    queries, query_info = build_sibling_queries(
        embeddings, train_idx, source_ids, args.n_sib, args.dirichlet, args.seed
    )
    query_groups = source_ids[train_idx]
    print({"training_captions": len(train_rows), **query_info}, flush=True)
    if args.operator == "full":
        retrieval = precompute_query_mix(
            queries, embeddings, source_ids, query_groups, device=str(device),
            forbidden=forbidden, chunk=args.retrieval_chunk or 128,
        )
    else:
        group_counts = Counter(group_ids[train_rows])
        effective_k = min(args.topk, len(train_rows) - max(group_counts.values()))
        retrieval = precompute_query_neighbors(
            queries, embeddings, source_ids, query_groups, k=effective_k,
            device=str(device), forbidden=forbidden, chunk=args.retrieval_chunk or 256,
        )
    del queries

    train_texts = [texts[i] for i in train_rows]
    tokens = torch.cat([
        clip.tokenize(train_texts[start:start + 256], truncate=True)[:, :MAX_LEN].clone()
        for start in range(0, len(train_texts), 256)
    ]).long()
    del train_texts, texts
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_rows))), batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed),
    )
    model = load_decoder(args.init_decoder, device)
    model.train()
    if args.mapper_only:
        model.requires_grad_(False)
        model.decoder.eval()
        model.clip_project.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    steps = len(loader) * args.epochs
    if args.max_steps is not None:
        steps = min(steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(1000, max(steps // 10, 1)), num_training_steps=steps
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    step = 0
    with tqdm(total=steps, desc="PoE training") as progress:
        for _ in range(args.epochs):
            for (rows,) in loader:
                target = tokens[rows].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if args.operator == "full":
                        prefix = retrieval["prefix"][rows].to(device)
                        prefix = torch.nn.functional.normalize(prefix, dim=-1)
                    else:
                        candidates = embeddings[retrieval["index"][rows]].to(device)
                        weights = retrieval["pi"][rows].to(device)
                        prefix = embed_mean(candidates, weights)
                    loss = -sequence_logprob(model, prefix.unsqueeze(1), target).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                if step % 200 == 0:
                    progress.set_postfix(loss=f"{loss.item():.4f}")
                progress.update()
                if step == steps:
                    break
            if step == steps:
                break

    with atomic_output(args.output, binary=True) as stream:
        torch.save(model.state_dict(), stream)


def main():
    parser = argparse.ArgumentParser(description="Train the PoE caption generator using grouped captions.")
    parser.add_argument("--corpus", choices=("coco", "flickr"), required=True)
    parser.add_argument("--captions", type=Path, required=True, help="Karpathy caption dataset JSON.")
    parser.add_argument("--memory", type=Path, required=True, help="COCO HDF5 bank or Flickr30k PT cache.")
    parser.add_argument("--init-decoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Ruggero1912/Patch-ioner_talk2dino_decap_COCO_Captions",
                        help="OnePatch model ID or local configuration for Flickr30k text encoding.")
    parser.add_argument("--text-groups", type=Path, help="Optional COCO text group IDs in NPY format.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--n-sib", type=int, help="Maximum same-image observations per query.")
    parser.add_argument("--dirichlet", action="store_true", help="Sample observation weights instead of averaging.")
    parser.add_argument("--operator", choices=("topk", "full"), default="topk")
    parser.add_argument("--mapper-only", action="store_true", help="Freeze GPT while adapting the PoE prefix mapper.")
    parser.add_argument("--retrieval-chunk", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    sizes = (args.batch_size, args.epochs, args.topk, args.retrieval_chunk, args.max_steps)
    if any(value is not None and value < 1 for value in sizes) or args.learning_rate <= 0:
        parser.error("Batch size, epochs, top-k, retrieval chunk, steps and learning rate must be positive.")
    if args.n_sib is not None and args.n_sib < 2:
        parser.error("--n-sib must be at least 2 for the PoE training path.")
    if args.corpus == "flickr" and args.text_groups is not None:
        parser.error("--text-groups applies to COCO memory only.")
    train(args)


if __name__ == "__main__":
    main()
