#!/usr/bin/env python3
"""
generate_dataset.py — Génère un dataset d'images de lettres pour l'entraînement CNN.

Structure de sortie :
    training_data/
        A/  0000.png  0001.png  …
        B/  …
        …
        Z/  …

Chaque image est en niveaux de gris, 28×28 pixels (attendu par le CNN).

Dépendances :
    pip install Pillow numpy scipy
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from scipy.ndimage import map_coordinates, gaussian_filter, grey_dilation, grey_erosion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGES_PER_LETTER  = 500
IMAGE_SIZE         = 56

FONT_SIZE_MIN      = 0.55   # lettre plus grande → moins de risque de décomposition
FONT_SIZE_MAX      = 0.85

ROTATION_MAX_DEG   = 8.0    # moins de rotation → évite les artefacts post-binarisation
SHEAR_MAX          = 0.10   # moins de cisaillement

MORPH_MAX_ITER     = 1      # morphologie plus légère

CONTRAST_MIN       = 0.85
CONTRAST_MAX       = 1.2
BRIGHTNESS_MIN     = 0.9
BRIGHTNESS_MAX     = 1.1

BLUR_RADIUS_MAX    = 0.8    # moins de flou → traits plus nets après binarisation

ELASTIC_ALPHA_MIN  = 0.0
ELASTIC_ALPHA_MAX  = 1.5    # déformation réduite — évite d'émietter les lettres
ELASTIC_SIGMA      = 5.0    # plus lisse = déformations plus cohérentes

NOISE_STD_MIN      = 1.0
NOISE_STD_MAX      = 8.0    # bruit réduit → binarisation plus propre

BG_BASE            = 245    # fond très clair, proche des grilles scannées
BG_VARIATION       = 8
FG_BASE            = 15     # encre noire dense
FG_VARIATION       = 10

OFFSET_FRAC        = 0.10

EXTRA_FONT_PATHS: list[str] = []

# ---------------------------------------------------------------------------
# Paramètres du filtre de polices (renforcé)
# ---------------------------------------------------------------------------

FILTER_TEST_SIZE      = 80     # taille de rendu pour les tests
FILTER_MIN_CONTRAST   = 70     # contraste min légèrement plus strict qu'avant (60)
FILTER_MIN_DARK_FRAC  = 0.03   # fraction minimale de pixels "encre"
FILTER_MIN_BBOX_RATIO = 0.25   # bbox height ≥ 25% de la font size
FILTER_MIN_UNIQUE     = 24     # au moins 24 glyphes distincts sur 26 lettres

# ---------------------------------------------------------------------------
# Recherche de polices TTF/OTF sur le système
# ---------------------------------------------------------------------------

def find_system_fonts() -> list[str]:
    """Retourne une liste de chemins vers des polices TTF/OTF disponibles."""
    fonts: list[str] = []

    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        "/Library/Fonts",
        "/System/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "C:/Windows/Fonts",
    ]

    for d in search_dirs:
        p = Path(d)
        if p.is_dir():
            for ext in ("*.ttf", "*.TTF", "*.otf", "*.OTF"):
                fonts.extend(str(f) for f in p.rglob(ext))

    for path in EXTRA_FONT_PATHS:
        if Path(path).is_file():
            fonts.append(path)

    seen: set[str] = set()
    unique: list[str] = []
    for f in fonts:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


def _render_char(ch: str, font: ImageFont.FreeTypeFont) -> tuple[np.ndarray, tuple]:
    """Rend un caractère centré et retourne (array, bbox)."""
    canvas_sz = FILTER_TEST_SIZE * 2
    img  = Image.new("L", (canvas_sz, canvas_sz), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    x = (canvas_sz - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (canvas_sz - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), ch, font=font, fill=0)
    return np.array(img), bbox


def load_fonts(font_paths: list[str]) -> list[str]:
    """
    Filtre strict des polices : garde uniquement celles qui contiennent
    les 26 lettres A–Z correctement rendues.

    4 critères appliqués sur TOUTES les lettres A–Z :
      1. Bounding box height ≥ FILTER_MIN_BBOX_RATIO × font_size
         → élimine les glyphes vides/invisibles (lettre absente de la police)
      2. Contraste (max - min) ≥ FILTER_MIN_CONTRAST
         → élimine les traits fantômes quasi-invisibles
      3. Fraction de pixels "encre" ≥ FILTER_MIN_DARK_FRAC
         → élimine les polices ultra-fines illisibles à 28×28
      4. Au moins FILTER_MIN_UNIQUE glyphes distincts sur 26
         → élimine les polices qui remplacent les lettres manquantes
           par un même glyphe de substitution (tofu □)
    """
    usable: list[str] = []
    all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for path in font_paths:
        try:
            font = ImageFont.truetype(path, FILTER_TEST_SIZE)
            ok   = True
            glyph_hashes: set[bytes] = set()

            for ch in all_letters:
                arr, bbox = _render_char(ch, font)

                # Critère 1 : bounding box non négligeable
                bbox_h = bbox[3] - bbox[1]
                if bbox_h < FILTER_MIN_BBOX_RATIO * FILTER_TEST_SIZE:
                    ok = False
                    break

                # Critère 2 : contraste suffisant
                contrast = int(arr.max()) - int(arr.min())
                if contrast < FILTER_MIN_CONTRAST:
                    ok = False
                    break

                # Critère 3 : surface encrée suffisante
                threshold = arr.min() + contrast // 2
                dark_frac = (arr < threshold).mean()
                if dark_frac < FILTER_MIN_DARK_FRAC:
                    ok = False
                    break

                # Critère 4 : unicité du glyphe (accumulation)
                # On hash le contenu de la bounding box pour détecter
                # les glyphes identiques (substitution par tofu)
                x0 = max(0, bbox[0])
                y0 = max(0, bbox[1])
                x1 = min(arr.shape[1], bbox[2])
                y1 = min(arr.shape[0], bbox[3])
                crop = arr[y0:y1, x0:x1]
                glyph_hashes.add(crop.tobytes())

            if not ok:
                continue

            # Critère 4 (final) : nombre de glyphes distincts
            if len(glyph_hashes) < FILTER_MIN_UNIQUE:
                continue

            usable.append(path)

        except Exception:
            pass

    return usable


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def apply_shear(canvas: Image.Image, bg: int) -> Image.Image:
    """Cisaillement affine horizontal et/ou vertical aléatoire."""
    shx = random.uniform(-SHEAR_MAX, SHEAR_MAX)
    shy = random.uniform(-SHEAR_MAX, SHEAR_MAX)
    w, h = canvas.size
    cx, cy = w / 2, h / 2
    a, b = 1.0, shx
    d, e = shy, 1.0
    c = cx - a * cx - b * cy
    f = cy - d * cx - e * cy
    return canvas.transform(
        (w, h), Image.AFFINE, (a, b, c, d, e, f),
        resample=Image.BICUBIC, fillcolor=bg
    )


def apply_morphology(canvas: Image.Image, bg: int, fg: int) -> Image.Image:
    """
    Dilatation ou érosion morphologique en NIVEAUX DE GRIS.
    Utilise grey_dilation/grey_erosion pour préserver l'antialiasing PIL
    (l'ancienne version avec binary_dilation reconstruisait une image 2 niveaux).
    """
    n_iter = random.randint(0, MORPH_MAX_ITER)
    if n_iter == 0:
        return canvas

    arr = np.array(canvas, dtype=np.uint8)
    op  = random.choice(["dilate", "erode"])

    new_arr = arr.copy()
    for _ in range(n_iter):
        # Fond clair / encre sombre → on inverse avant grey_dilation
        # pour que l'opération agrandisse bien l'encre (valeurs basses)
        inv = 255 - new_arr
        if op == "dilate":
            inv = grey_dilation(inv, size=(3, 3))
        else:
            inv = grey_erosion(inv, size=(3, 3))
        new_arr = 255 - inv

    # Sécurité : si l'érosion a trop effacé la lettre, on annule
    if op == "erode":
        orig_ink = (arr    < (bg + fg) // 2).sum()
        new_ink  = (new_arr < (bg + fg) // 2).sum()
        if orig_ink > 0 and new_ink < orig_ink * 0.3:
            return canvas

    return Image.fromarray(new_arr, mode="L")


def apply_elastic(arr: np.ndarray) -> np.ndarray:
    """
    Distorsion élastique légère (Simard et al., 2003).
    Utilise np.random.uniform (respecte np.random.seed) au lieu de
    np.random.default_rng() qui crée un générateur non seeded indépendant.
    """
    alpha = random.uniform(ELASTIC_ALPHA_MIN, ELASTIC_ALPHA_MAX)
    if alpha < 0.5:
        return arr

    h, w = arr.shape
    dx = gaussian_filter(np.random.uniform(-1, 1, (h, w)), sigma=ELASTIC_SIGMA) * alpha
    dy = gaussian_filter(np.random.uniform(-1, 1, (h, w)), sigma=ELASTIC_SIGMA) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = [
        np.clip(y + dy, 0, h - 1).ravel(),
        np.clip(x + dx, 0, w - 1).ravel(),
    ]
    return map_coordinates(arr, coords, order=1, mode="nearest").reshape(h, w)


# ---------------------------------------------------------------------------
# Génération d'une image
# ---------------------------------------------------------------------------

def render_letter(letter: str, font_path: str) -> Image.Image:
    """
    Génère une image 28×28 en niveaux de gris d'une lettre avec toutes
    les augmentations activées aléatoirement.
    """
    render_size = IMAGE_SIZE * 4  # haute résolution, réduit à la fin

    font_size = int(render_size * random.uniform(FONT_SIZE_MIN, FONT_SIZE_MAX))
    font = ImageFont.truetype(font_path, font_size)

    bg = max(200, min(255, int(random.gauss(BG_BASE, BG_VARIATION))))
    fg = max(0,   min(60,  int(random.gauss(FG_BASE, FG_VARIATION))))

    canvas = Image.new("L", (render_size, render_size), color=bg)
    draw   = ImageDraw.Draw(canvas)

    bbox   = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    max_offset = int(render_size * OFFSET_FRAC)
    ox = random.randint(-max_offset, max_offset)
    oy = random.randint(-max_offset, max_offset)

    x = (render_size - tw) // 2 - bbox[0] + ox
    y = (render_size - th) // 2 - bbox[1] + oy
    draw.text((x, y), letter, font=font, fill=fg)

    # Rotation
    angle = random.uniform(-ROTATION_MAX_DEG, ROTATION_MAX_DEG)
    canvas = canvas.rotate(angle, resample=Image.BICUBIC, expand=False,
                           fillcolor=bg)

    # Cisaillement
    canvas = apply_shear(canvas, bg)

    # Morphologie niveaux de gris
    canvas = apply_morphology(canvas, bg, fg)

    # Contraste et luminosité
    canvas = ImageEnhance.Contrast(canvas).enhance(
        random.uniform(CONTRAST_MIN, CONTRAST_MAX))
    canvas = ImageEnhance.Brightness(canvas).enhance(
        random.uniform(BRIGHTNESS_MIN, BRIGHTNESS_MAX))

    # Flou gaussien léger
    blur_r = random.uniform(0.0, BLUR_RADIUS_MAX)
    if blur_r > 0.1:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # Réduction à IMAGE_SIZE × IMAGE_SIZE
    canvas = canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    arr = np.array(canvas, dtype=np.float32)

    # Distorsion élastique
    arr = apply_elastic(arr)

    # Bruit gaussien
    arr += np.random.normal(0.0, random.uniform(NOISE_STD_MIN, NOISE_STD_MAX), arr.shape)
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr_u8, mode="L")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def generate_dataset(output_dir: str, n_per_letter: int,
                     font_paths: list[str]) -> None:
    letters = [chr(ord('A') + i) for i in range(26)]
    os.makedirs(output_dir, exist_ok=True)

    for letter in letters:
        letter_dir = os.path.join(output_dir, letter)
        os.makedirs(letter_dir, exist_ok=True)

        print(f"  Génération {letter} ({n_per_letter} images)…", end="", flush=True)
        for i in range(n_per_letter):
            font_path = random.choice(font_paths)
            img = render_letter(letter, font_path)
            img.save(os.path.join(letter_dir, f"{i:04d}.png"))
        print(" OK")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Génère le dataset d'entraînement pour le CNN OCR."
    )
    parser.add_argument(
        "-n", "--count", type=int, default=IMAGES_PER_LETTER,
        help=f"Images par lettre (défaut : {IMAGES_PER_LETTER})"
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Répertoire de sortie (défaut : dossier du script)"
    )
    args = parser.parse_args()

    print("Recherche des polices système…")
    all_font_paths = find_system_fonts()
    print(f"  {len(all_font_paths)} police(s) trouvée(s) sur le système.")

    print("Validation des polices (filtre strict A–Z)…")
    usable = load_fonts(all_font_paths)
    print(f"  {len(usable)}/{len(all_font_paths)} police(s) validée(s).")

    if not usable:
        print("ERREUR : aucune police TTF/OTF utilisable trouvée.", file=sys.stderr)
        print("Solutions :", file=sys.stderr)
        print("  sudo apt install fonts-dejavu fonts-liberation fonts-freefont-ttf",
              file=sys.stderr)
        print("  ou ajoutez des chemins dans EXTRA_FONT_PATHS.", file=sys.stderr)
        sys.exit(1)

    for p in usable:
        print(f"  ✓ {Path(p).name}")

    random.seed(42)
    np.random.seed(42)

    print(f"\nGénération du dataset dans '{args.output}' "
          f"({args.count} images × 26 lettres = {args.count * 26} images)…")
    generate_dataset(args.output, args.count, usable)

    print("\nDataset généré avec succès.")


if __name__ == "__main__":
    main()
