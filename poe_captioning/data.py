import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


TRAIN_SPLITS = ("train", "restval")
EVAL_SPLITS = ("val", "test")


def json_load(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


@contextmanager
def atomic_output(path, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb" if binary else "w", encoding=None if binary else "utf-8",
                                dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def load_karpathy(path: Path) -> dict[str, list[dict]]:
    payload = json_load(path)
    images = payload["images"] if isinstance(payload, dict) else payload
    splits: dict[str, list[dict]] = defaultdict(list)
    for item in images:
        split = str(item.get("split") or "")
        filename = item.get("filename") or item.get("file_name") or ""
        image_id = int(item.get("cocoid") or item.get("imgid") or 0)
        captions = []
        for sent in item.get("sentences") or []:
            text = sent.get("raw") if isinstance(sent, dict) else str(sent)
            text = " ".join(str(text).strip().split())
            if text:
                captions.append(text)
        filepath = "train2014" if "train" in str(filename) else "val2014"
        splits[split].append(
            {
                "image_id": image_id,
                "filename": filename,
                "filepath": filepath,
                "split": split,
                "references": captions,
            }
        )
    return dict(splits)


def load_flickr_karpathy(path: Path) -> dict[str, list[dict]]:
    payload = json_load(path)
    images = payload["images"] if isinstance(payload, dict) else payload
    splits: dict[str, list[dict]] = defaultdict(list)
    for item in images:
        split = str(item.get("split") or "")
        filename = str(item.get("filename") or item.get("file_name") or "")
        stem = Path(filename).stem
        image_id = int(stem) if stem.isdigit() else int(item.get("imgid") or 0)
        captions = []
        for sent in item.get("sentences") or []:
            text = sent.get("raw") if isinstance(sent, dict) else str(sent)
            text = " ".join(str(text).strip().split())
            if text:
                captions.append(text)
        splits[split].append(
            {
                "image_id": image_id,
                "filename": filename,
                "split": split,
                "references": captions,
            }
        )
    return dict(splits)


def caption_index(karpathy: dict[str, list[dict]]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for split, samples in karpathy.items():
        for sample in samples:
            image_id = int(sample["image_id"])
            for cap in sample["references"]:
                key = normalize_text(cap)
                rec = index.setdefault(key, {"splits": set(), "image_ids": set()})
                rec["splits"].add(split)
                rec["image_ids"].add(image_id)
    return index


def assign_memory_rows(texts: list[str], index: dict[str, dict]) -> dict:
    n = len(texts)
    image_ids = np.full(n, -1, dtype=np.int64)
    split_names = []
    train_idx = []
    eval_idx = []
    unmatched_idx = []
    n_eval_overlap = 0
    eval_set = set(EVAL_SPLITS)
    train_set = set(TRAIN_SPLITS)
    for i, text in enumerate(texts):
        rec = index.get(normalize_text(text))
        if rec is None:
            split_names.append("unmatched")
            unmatched_idx.append(i)
            continue
        splits = rec["splits"]
        ids = rec["image_ids"]
        if len(ids) == 1:
            image_ids[i] = int(next(iter(ids)))
        if splits & eval_set:
            split_names.append("eval")
            eval_idx.append(i)
            if splits & train_set:
                n_eval_overlap += 1
            continue
        if splits & train_set:
            split_names.append("train")
            train_idx.append(i)
            continue
        split_names.append("other")
        unmatched_idx.append(i)
        image_ids[i] = -1
    return {
        "image_ids": image_ids,
        "split_names": split_names,
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "eval_idx": np.asarray(eval_idx, dtype=np.int64),
        "unmatched_idx": np.asarray(unmatched_idx, dtype=np.int64),
        "n_eval_overlap": n_eval_overlap,
    }


def source_ids_for_neighbors(image_ids: np.ndarray, text_groups: np.ndarray) -> np.ndarray:
    src = np.asarray(image_ids, dtype=np.int64).copy()
    unknown = src < 0

    src[unknown] = 10_000_000 + np.asarray(text_groups, dtype=np.int64)[unknown]
    return src


def resolve_image(coco_root: Path, img_id) -> Path | None:
    stem = str(int(img_id)).zfill(12)
    for cand in (
        coco_root / "val2017" / f"{stem}.jpg",
        coco_root / "train2017" / f"{stem}.jpg",
        coco_root / "val2014" / f"COCO_val2014_{stem}.jpg",
        coco_root / "val2014" / f"{stem}.jpg",
        coco_root / "train2014" / f"COCO_train2014_{stem}.jpg",
        coco_root / "train2014" / f"{stem}.jpg",
    ):
        if cand.exists():
            return cand
    return None


def load_h5_memory(path: Path) -> tuple[np.ndarray, list[str]]:
    import h5py
    with h5py.File(path, "r") as handle:
        embs = handle["coco_captions-embeddings"][:]
        raw = handle["coco_captions-text"][:]
    texts = []
    for item in raw:
        if isinstance(item, (bytes, np.bytes_)):
            texts.append(item.decode("utf-8", "ignore"))
        else:
            texts.append(str(item))
    return embs.astype("float32"), texts


def coco_memory_groups(texts, annotations, text_groups=None):
    index = caption_index(load_karpathy(annotations))
    assignment = assign_memory_rows(texts, index)
    train_idx = assignment["train_idx"]
    if not len(train_idx):
        raise ValueError("No training captions match the COCO memory.")
    if text_groups is None:
        _, groups = np.unique([normalize_text(text) for text in texts], return_inverse=True)
    else:
        groups = np.load(text_groups, allow_pickle=False)
        if groups.shape != (len(texts),):
            raise ValueError("Text group IDs must align with COCO memory rows.")
    source_ids = source_ids_for_neighbors(assignment["image_ids"], groups)
    return train_idx, source_ids


def flickr_train_memory(path, annotations, model_id, device):
    import torch

    rows = load_flickr_karpathy(annotations).get("train", [])
    texts = [text for row in rows for text in row["references"]]
    image_ids = [row["image_id"] for row in rows for _ in row["references"]]
    if not texts:
        raise ValueError("No Flickr30k training captions were found.")
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if list(payload["texts"]) != texts or np.asarray(payload["image_ids"]).tolist() != image_ids:
            raise ValueError("The Flickr30k memory does not match the supplied training split.")
        embeddings = torch.as_tensor(payload["embs"], dtype=torch.float32)
    else:
        from .backend import encode_texts_official, load_patchioner

        model = load_patchioner(model_id, device)
        embeddings = encode_texts_official(model, texts, device, batch_size=256)
        del model
        with atomic_output(path, binary=True) as stream:
            torch.save({"embs": embeddings, "texts": texts, "image_ids": image_ids}, stream)
    if embeddings.ndim != 2 or embeddings.shape != (len(texts), 768):
        raise ValueError("The Flickr30k embeddings must have shape (number of captions, 768).")
    return embeddings, texts, np.asarray(image_ids, dtype=np.int64)


def load_evaluation_rows(task, corpus, split, annotations, image_root):
    rows = []
    if task == "image":
        loader = load_karpathy if corpus == "coco" else load_flickr_karpathy
        for image in loader(annotations).get(split, []):
            filename = Path(image["filename"]).name
            path = image_root / image.get("filepath", "") / filename
            if not path.is_file():
                path = image_root / filename
            if not path.is_file() and corpus == "coco":
                path = resolve_image(image_root, image["image_id"])
            if path is None or not path.is_file():
                raise FileNotFoundError(filename)
            if image["references"]:
                rows.append({"image_id": image["image_id"], "path": path, "refs": image["references"]})
    else:
        for image_id, annotation in json_load(annotations).items():
            if corpus == "coco":
                path = resolve_image(image_root, image_id)
            else:
                candidates = [image_root / f"{image_id}{suffix}" for suffix in (".jpg", ".JPG", ".jpeg")]
                path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Image ID {image_id} was not found under the image root.")
            if task == "trace":
                for caption, trace in zip(annotation["captions"], annotation["traces"]):
                    rows.append({"image_id": image_id, "path": path, "refs": [caption], "trace": trace})
            elif task == "region_set":
                for caption, boxes in annotation.items():
                    xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
                    xywh = [box for box in xywh if box[2] > 1 and box[3] > 1]
                    if xywh:
                        rows.append({"image_id": image_id, "path": path, "refs": [caption], "bboxes": xywh})
            else:
                raise ValueError(task)
    if not rows:
        raise ValueError("No evaluation rows were found for the requested task and split.")
    return rows
