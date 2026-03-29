# OCR & Résolveur de mots croisés

Projet EPITA S3 2028 — Groupe 3, TLS.

OCR complet en C (C99) avec SDL2 et libpng, capable de lire les lettres
d'une grille de mots croisés et d'y rechercher des mots.

---

## Pipeline

```
Image PNG
  └─► Prétraitement     grayscale → binarisation → (rotation optionnelle)
        └─► Segmentation    détection des cellules de la grille
              └─► CNN         reconnaissance de chaque lettre (A–Z)
                    └─► Résolveur   recherche de mots en 8 directions
```

---

## Dépendances

| Bibliothèque | Rôle |
|---|---|
| **libpng** | Chargement et sauvegarde des images PNG |
| **SDL2** | Rendu (rotation d'image) |
| **libm** | Fonctions mathématiques (sqrt, cos, …) |
| **libpthread** | Parallélisme lors du chargement du dataset |

Installation sur Arch Linux :
```bash
sudo pacman -S sdl2 libpng
```

Installation sur Debian/Ubuntu :
```bash
sudo apt install libsdl2-dev libpng-dev
```

---

## Compilation

```bash
make all      # compile ./train et ./solve  (cible par défaut)
make train    # compile uniquement ./train
make solve    # compile uniquement ./solve
make clean    # supprime les .o et les binaires
```

Option debug :
```bash
make DEBUG=1
```

---

## Utilisation

### Entraînement

```bash
./train --data training_data/ --output models/mon_modele.bin -j4
```

| Option | Défaut | Description |
|---|---|---|
| `--data <dossier>` | `training_data/` | Racine du dataset (sous-dossiers A–Z) |
| `--output <fichier>` | `models/model_<ts>.bin` | Fichier de sortie du modèle |
| `-j<N>` | `1` | Nombre de threads de chargement |

Structure attendue du dataset :
```
training_data/
  A/  img1.png  img2.png  …
  B/  …
  …
  Z/  …
```

### Résolution

```bash
./solve --image grille.png --model models/mon_modele.bin --words CHAT,CHIEN -v
```

| Option | Défaut | Description |
|---|---|---|
| `--image <fichier>` | *(obligatoire)* | Image de la grille à analyser |
| `--model <fichier>` | dernier `.bin` dans `models/` | Modèle à utiliser |
| `--words <liste>` | *(aucune)* | Mots à rechercher, séparés par des virgules |
| `--verbose` / `-v` | désactivé | Affiche les détails de reconnaissance |

Si aucun `--model` n'est fourni et que le dossier `models/` est vide,
`./solve` affiche une erreur claire et quitte avec le code `2`.

---

## Architecture CNN

```
Entrée 28×28 (float, 0–1)
  Conv2D  16 filtres 3×3        →  16×26×26
  ReLU
  MaxPool 2×2                   →  16×13×13
  Flatten                       →  2704
  Dense   2704 → 128
  ReLU
  Dense    128 →  26
  Softmax                       →  P(A) … P(Z)
```

Entraînement : initialisation He, SGD + momentum (lr = 0,001 ; β = 0,9),
perte cross-entropie, taille de batch 32.

---

## Structure du projet

```
.
├── Makefile
├── train_main.c            Point d'entrée de ./train
├── solve_main.c            Point d'entrée de ./solve
├── src/
│   ├── cnn/
│   │   ├── cnn.h / cnn.c          Architecture CNN, passe avant/arrière
│   │   ├── model.h / model.c      Sérialisation binaire du modèle
│   │   └── dataset.h / dataset.c  Chargement du dataset (POSIX threads)
│   ├── preprocess/
│   │   └── image.h / image.c      Chargement libpng, grayscale, binarisation, resize
│   ├── segment/
│   │   └── segment.h / segment.c  Composantes connexes, détection de grille
│   └── solver/
│       └── solver.h / solver.c    Recherche en 8 directions
├── models/                 Modèles entraînés (.bin)
├── training_data/          Dataset d'entraînement (A–Z)
└── CLAUDE.md               Suivi des tâches et documentation interne
```

### Modules hérités (référence)

| Dossier | Contenu |
|---|---|
| `OCR-project/` | Implémentation modulaire précédente |
| `epita-prepa-computer-science-proj-s3-2028-tls-groupe3/` | Prototype initial du groupe |

---

## Format du fichier modèle

```
[magic 4 octets : 'OCRC'] [version uint32] [CNNWeights : ~1,4 Mo]
```

Le fichier est un dump binaire plat de la structure `CNNWeights`.
Version actuelle : **1**.

---

## Codes de retour

| Binaire | Code | Signification |
|---|---|---|
| `train` | `0` | Succès |
| `train` | `1` | Erreur d'argument |
| `train` | `2` | Erreur de chargement du dataset |
| `train` | `3` | Erreur de sauvegarde du modèle |
| `solve` | `0` | Succès |
| `solve` | `1` | Erreur d'argument |
| `solve` | `2` | Modèle introuvable ou invalide |
| `solve` | `3` | Erreur de chargement de l'image |
| `solve` | `4` | Erreur de segmentation |
