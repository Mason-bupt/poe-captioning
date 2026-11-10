import torch
import torch.nn.functional as F
from PIL import Image

PREFIX_SIZE = 768
MAX_LEN = 30


def load_patchioner(model_id, device):
    from patchioner import Patchioner

    model = Patchioner.from_config(str(model_id), device=str(device))
    model.eval()
    return model


def load_decoder(checkpoint, device):
    from patchioner.decap.decap import get_decap_model

    return get_decap_model(device, str(checkpoint), prefix_size=PREFIX_SIZE)


def dino_forward(model, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        outs = model.dino(pixels, is_training=True)
    return {
        "cls": outs["x_norm_clstoken"],
        "patches": outs["x_norm_patchtokens"],
        "registers": outs.get("x_norm_regtokens"),
    }


def avg_self_attn_token(model, pixels: torch.Tensor, patches: torch.Tensor | None = None) -> torch.Tensor:
    from patchioner.dino_extraction import feats, process_self_attention

    if patches is None:
        patches = dino_forward(model, pixels)["patches"]
    if "self_attn" not in feats:
        raise RuntimeError("DINO self-attention hook did not fire; cannot match avg_self_attn")
    self_attn, _ = process_self_attention(
        feats["self_attn"],
        pixels.shape[0],
        model.num_tokens,
        model.num_attn_heads,
        model.embed_dim,
        model.scale,
        model.num_global_tokens,
        ret_self_attn_maps=True,
    )
    return (self_attn.unsqueeze(-1) * patches).mean(dim=1)


def trace_visual(model, pixels: torch.Tensor, traces: list) -> tuple[torch.Tensor, torch.Tensor]:
    from patchioner.bbox_utils import map_traces_to_grid

    patches = dino_forward(model, pixels)["patches"]
    n_grid = int(patches.shape[1] ** 0.5)
    dim = patches.shape[-1]
    grids = torch.stack([map_traces_to_grid(tr, n_grid) for tr in traces], dim=0).to(pixels.device)
    if grids.shape[0] != pixels.shape[0]:
        patches = patches[:1].expand(grids.shape[0], -1, -1)
        reshaped = patches.reshape(grids.shape[0], n_grid, n_grid, dim)
    else:
        reshaped = patches.reshape(pixels.shape[0], n_grid, n_grid, dim)
    vis = (grids.unsqueeze(-1) * reshaped).mean(dim=(1, 2))
    return vis, patches


def bbox_visual(model, pixels: torch.Tensor, bboxes: torch.Tensor, single_per_image: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    from patchioner.bbox_utils import extract_bboxes_feats

    patches = dino_forward(model, pixels)["patches"]
    feats = extract_bboxes_feats(
        patches,
        bboxes,
        gaussian_avg=False,
        patch_size=model.patch_size,
        get_single_embedding_per_image=single_per_image,
    )
    if feats.ndim > 2:
        feats = feats.reshape(-1, feats.shape[-1])
    return feats, patches


def encode_texts_official(model, texts: list[str], device, batch_size: int = 64) -> torch.Tensor:
    import clip

    proj = model.im_proj
    if getattr(proj, "clip_model", None) is None or not hasattr(proj, "talk2dino"):
        proj._load_models()
    clip_model = proj.clip_model
    talk2dino = proj.talk2dino
    outs = []
    clip_model.eval()
    talk2dino.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device).long()
            feat = clip_model.encode_text(tokens).float()
            feat = talk2dino.project_clip_txt(feat)
            feat = F.normalize(feat, dim=-1)
            outs.append(feat.cpu())
    return torch.cat(outs, dim=0) if outs else torch.zeros(0, PREFIX_SIZE)


def _prefix_emb(decap, embs_raw: torch.Tensor) -> torch.Tensor:
    x = F.normalize(embs_raw.float(), dim=-1)
    z = decap.clip_project(x)
    return z.view(embs_raw.shape[0], 1, -1)


def sequence_logprob(decap, embs_raw: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    if embs_raw.ndim == 2:
        embs_raw = embs_raw.unsqueeze(0)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    tokens = tokens.long()
    b, k, _ = embs_raw.shape
    t = tokens.shape[1]
    z = _prefix_emb(decap, embs_raw.reshape(b * k, -1)).view(b, k, 1, -1)
    layer = decap.decoder.get_input_embeddings()
    text = layer(tokens).unsqueeze(1).expand(b, k, t, -1)
    cat = torch.cat([z, text], dim=2).reshape(b * k, t + 1, -1)
    out = decap.decoder(inputs_embeds=cat, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    logp = F.log_softmax(logits, dim=-1)
    tgt = tokens.unsqueeze(1).expand(b, k, t).reshape(b * k, t)
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    nll = nll.masked_fill(tgt.eq(0), 0.0)
    return (-nll.sum(dim=-1)).view(b, k)


def visual_queries(model, rows, task, device):
    transforms = model.image_transforms_no_crop
    pixels = []
    boxes = []
    if task == "region_set":
        from patchioner.bbox_utils import adjust_bbox_for_transform_no_scale

    for row in rows:
        with Image.open(row["path"]) as handle:
            image = handle.convert("RGB")
            pixels.append(transforms(image))
            if task == "region_set":
                boxes.append([
                    adjust_bbox_for_transform_no_scale(image, box, model.resize_dim, model.resize_dim)
                    for box in row["bboxes"]
                ])

    pixels = torch.stack(pixels).to(device)
    if task == "image":
        return avg_self_attn_token(model, pixels)
    if task == "trace":
        return trace_visual(model, pixels, [row["trace"] for row in rows])[0]
    if task == "region_set":
        max_boxes = max(map(len, boxes))
        padded = [row + [[-1, -1, -1, -1]] * (max_boxes - len(row)) for row in boxes]
        tensor = torch.tensor(padded, dtype=torch.float32, device=device)
        return bbox_visual(model, pixels, tensor, single_per_image=True)[0]
    raise ValueError(task)


def replace_memory(model, embeddings, texts):
    original = model.im_proj.embs_dataset
    device = original.device if torch.is_tensor(original) else torch.device("cpu")
    model.im_proj.embs_dataset = torch.as_tensor(embeddings, dtype=torch.float32, device=device).contiguous()
    model.im_proj.text_dataset = list(texts)


def generate(model, queries):
    captions = model.caption_tokens(queries, project=True, decoding_strategy="greedy")
    if isinstance(captions, tuple):
        captions = captions[0]
    if isinstance(captions, str):
        captions = [captions]
    cleaned = []
    for caption in captions:
        if isinstance(caption, (list, tuple)):
            caption = caption[0]
        text = str(caption or "").replace("<|startoftext|>", "").replace("<|endoftext|>", "")
        cleaned.append(text.strip(" .").strip())
    if len(cleaned) != len(queries):
        raise RuntimeError("The decoder did not return one caption per visual query.")
    return cleaned
