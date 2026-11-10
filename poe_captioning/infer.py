import argparse
import json
from pathlib import Path


def infer(args):
    import torch
    from tqdm import tqdm

    from .backend import generate, load_decoder, load_patchioner, replace_memory, visual_queries
    from .data import atomic_output, coco_memory_groups, flickr_train_memory, load_evaluation_rows, load_h5_memory

    annotations = args.captions if args.task == "image" else args.eval_annotations
    rows = load_evaluation_rows(args.task, args.image_corpus or args.corpus, args.split, annotations, args.images)
    if args.limit is not None:
        rows = rows[:args.limit]

    device = torch.device(args.device)
    model = load_patchioner(args.model, device)
    if args.corpus == "coco":
        embeddings, texts = load_h5_memory(args.memory)
        train_idx, _ = coco_memory_groups(texts, args.captions)
        replace_memory(model, embeddings[train_idx], [texts[i] for i in train_idx])
    else:
        if not args.memory.is_file():
            raise FileNotFoundError("The Flickr30k training memory must be prepared before inference.")
        embeddings, texts, _ = flickr_train_memory(args.memory, args.captions, args.model, device)
        replace_memory(model, embeddings, texts)
    del embeddings, texts

    model.decoder = load_decoder(args.decoder, device)
    model.eval()
    batch_size = args.batch_size or (4 if args.task == "image" else 8)
    results = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size), desc=f"PoE {args.task}"):
            batch = rows[start:start + batch_size]
            queries = visual_queries(model, batch, args.task, device)
            captions = generate(model, queries)
            for index, (row, caption) in enumerate(zip(batch, captions, strict=True), start=start):
                results.append({
                    "unit_id": index,
                    "image_id": row["image_id"],
                    "refs": row["refs"],
                    "caption": caption,
                })

    with atomic_output(args.output) as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate captions using a PoE-trained decoder.")
    parser.add_argument("--task", choices=("image", "trace", "region_set"), required=True)
    parser.add_argument("--corpus", choices=("coco", "flickr"), required=True)
    parser.add_argument("--image-corpus", choices=("coco", "flickr"), help="Image corpus for a regional test.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Karpathy whole-image split.")
    parser.add_argument("--captions", type=Path, required=True, help="Karpathy caption dataset JSON.")
    parser.add_argument("--eval-annotations", type=Path, help="OnePatch trace or region-set test annotations.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True, help="COCO HDF5 bank or Flickr30k PT cache.")
    parser.add_argument("--decoder", type=Path, required=True, help="Checkpoint produced by PoE training.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Ruggero1912/Patch-ioner_talk2dino_decap_COCO_Captions",
                        help="OnePatch model ID or local Talk2DINO and DeCap configuration.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, help="Maximum evaluation units to decode.")
    args = parser.parse_args()
    if args.task != "image" and args.eval_annotations is None:
        parser.error("--eval-annotations is required for trace and region-set captioning.")
    if args.task != "image" and (args.corpus != "coco" or args.split != "test"):
        parser.error("Trace and region-set inference use the COCO decoder and memory on test annotations.")
    if args.task == "image" and args.image_corpus is not None:
        parser.error("--image-corpus applies to regional inference only.")
    if any(value is not None and value < 1 for value in (args.batch_size, args.limit)):
        parser.error("Batch size and limit must be positive.")
    infer(args)


if __name__ == "__main__":
    main()
