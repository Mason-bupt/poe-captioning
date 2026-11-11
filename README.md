PoE Captioning
==============

This repository contains PoE training on grouped captions and visual inference. Run the commands below from the repository root with Python 3.10 or newer. A CUDA device is recommended for full-memory retrieval.

Install [OnePatch](https://github.com/Ruggero1912/Patch-ioner) for the encoder, decoder, and visual pooling utilities.

```bash
python -m pip install git+https://github.com/Ruggero1912/Patch-ioner
```

Download `coco_karpathy-009.pt` and `coco_captions_text_embeddings-B16-ViT-B.16-591753.h5` from the public [Talk2DINO–DeCap model](https://huggingface.co/Ruggero1912/Patch-ioner_talk2dino_decap_COCO_Captions) into `assets/`. Obtain `dataset_coco.json` and `dataset_flickr30k.json` from the [Karpathy splits](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). Keep COCO images in the standard split directories under `data/coco/` and Flickr30k JPEGs in `data/flickr30k/`. For an offline installation, pass `--model` with a local Talk2DINO–DeCap configuration.

Training
--------

```bash
COCO_MEMORY=assets/coco_captions_text_embeddings-B16-ViT-B.16-591753.h5
DECODER_CHECKPOINT=assets/coco_karpathy-009.pt

python -m poe_captioning.train --corpus coco --captions data/dataset_coco.json \
  --memory "$COCO_MEMORY" --init-decoder "$DECODER_CHECKPOINT" --output checkpoints/poe_coco.pt

python -m poe_captioning.train --corpus flickr --captions data/dataset_flickr30k.json \
  --memory assets/flickr_train.pt --init-decoder "$DECODER_CHECKPOINT" --output checkpoints/poe_flickr.pt
```

The default recipe uses all available same-image siblings and excludes all entries from the target group during retrieval. Singleton groups use a self-query fallback, and the trainer reports their count. COCO uses Karpathy train and restval captions, with captions also found in val or test excluded. Flickr30k encodes the Karpathy train captions into the specified cache when needed. Use a trusted cache that corresponds to the supplied split and text encoder.

Defaults are one epoch, AdamW, batch size 32, learning rate 2e-5, weight decay 0.01, and top-8 retrieval with temperature 0.01. Both the prefix mapper and GPT are updated by default. Optional settings include `--n-sib`, `--dirichlet`, `--operator full`, `--mapper-only`, and `--seed`. Reduce `--retrieval-chunk` if the retrieval pass exceeds GPU memory.

Inference
---------

Inference uses the trained PoE checkpoint, the frozen visual encoder, full training-caption memory, and greedy decoding. The output JSON includes one entry per evaluation unit.

```bash
python -m poe_captioning.infer --task image --corpus coco \
  --captions data/dataset_coco.json --images data/coco --memory "$COCO_MEMORY" \
  --decoder checkpoints/poe_coco.pt --output predictions/coco_image.json

python -m poe_captioning.infer --task image --corpus flickr \
  --captions data/dataset_flickr30k.json --images data/flickr30k --memory assets/flickr_train.pt \
  --decoder checkpoints/poe_flickr.pt --output predictions/flickr_image.json

python -m poe_captioning.infer --task trace --corpus coco \
  --captions data/dataset_coco.json --images data/coco --memory "$COCO_MEMORY" \
  --eval-annotations data/trace_capt_coco_test.json \
  --decoder checkpoints/poe_coco.pt --output predictions/coco_trace.json

python -m poe_captioning.infer --task region_set --corpus coco \
  --captions data/dataset_coco.json --images data/coco --memory "$COCO_MEMORY" \
  --eval-annotations data/coco_entities_test.json \
  --decoder checkpoints/poe_coco.pt --output predictions/coco_region_set.json
```

The trace and region-set annotation files are available in the OnePatch repository under `Patch-ioner/eval-trace-captioning` and `Patch-ioner/eval-region-set-captioning`. Supply `--image-corpus flickr` with the corresponding annotations and Flickr30k image directory for the regional cross-dataset setting, which uses the COCO-adapted checkpoint and memory.
