# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

OCR and crossword puzzle solver, written in C (C99) with SDL2.
EPITA preparatory computer science project — Group 3, S3 2028.

**New unified pipeline:**
```
Input image
  → [src/preprocess] rotate / grayscale / binarize / resize 28×28
  → [src/segment]    detect grid cells, extract letter bounding boxes
  → [src/cnn]        CNN forward pass → predicted character (A–Z)
  → [src/solver]     8-direction crossword word search
```

**Two binaries:**
- `./train`  — load dataset, train CNN, save model
- `./solve`  — load model, run full OCR pipeline on an image

## Build

```bash
make all      # build both binaries (default)
make train    # build only ./train
make solve    # build only ./solve
make clean    # remove *.o and binaries
```

**Dependencies:** `libsdl2-dev`, `libsdl2-image-dev`, `libpthread` (system).

## Usage

```bash
# Train
./train --data training_data/ --output my_model.bin -j4

# Solve
./solve --image crossword.png --model my_model.bin --verbose
./solve --image crossword.png          # auto-picks latest model in models/
```

## Source layout

```
src/
├── cnn/
│   ├── cnn.h / cnn.c          CNN architecture + forward/backward
│   ├── model.h / model.c      binary save/load of weights
│   └── dataset.h / dataset.c  load training_data/A/ … Z/
├── preprocess/
│   └── image.h / image.c      SDL2 image load, rotate, binarize, resize
├── segment/
│   └── segment.h / segment.c  connected-component grid detection
└── solver/
    └── solver.h / solver.c    8-direction crossword solver
train_main.c                   entry point for ./train
solve_main.c                   entry point for ./solve
```

## CNN Architecture

```
Input 56×56 (float, 0–1)
  Conv2D  16 filters 3×3  → 16×54×54
  ReLU
  MaxPool 4×4             → 16×13×13
  Flatten                 → 2704
  Dense   2704 → 128
  ReLU
  Dense   128  → 26
  Softmax                 → P(A)…P(Z)
```

Training: He init, SGD + momentum (lr=0.001, β=0.9), averaged gradients,
cross-entropy loss, batch size 32.

## Task Checklist

- [x] Update CLAUDE.md with architecture and task list
- [x] Create src/ directory tree
- [x] src/cnn/cnn.h — structures, constants, declarations
- [x] src/cnn/cnn.c — forward pass, backward pass, He init, SGD
- [x] src/cnn/model.h / model.c — binary serialization
- [x] src/cnn/dataset.h / dataset.c — training_data/ loader
- [x] src/preprocess/image.h / image.c — SDL2 preprocessing pipeline
- [x] src/segment/segment.h / segment.c — letter segmentation
- [x] src/solver/solver.h / solver.c — crossword solver
- [x] train_main.c — CLI entry point for training
- [x] solve_main.c — CLI entry point for OCR pipeline
- [x] Makefile — all / train / solve / clean

## Key Data Files

| Path | Purpose |
|------|---------|
| `training_data/<X>/` | PNG images of letter X used for training |
| `models/` | Directory where trained models are saved |
| `<name>.bin` | Serialized CNN weights (binary) |

## Legacy code (kept for reference)

| Directory | Notes |
|-----------|-------|
| `OCR-project/` | Previous modular implementation |
| `epita-prepa-computer-science-proj-s3-2028-tls-groupe3/` | Prototype |
