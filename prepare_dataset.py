"""
GenImage subset preparation script.

Source: shimei123/Genimage on HuggingFace
  - BigGAN.zip  (934 MB) : 6000 real + 6000 fake
  - glide.zip   (1.2 GB) : 6000 real + 6000 fake

Both zips have the flat structure:
  <generator>/0_real/<filename>
  <generator>/1_fake/<filename>

Output layout:
  data/
    train/real/  fake/    <- from glide (train source)
    val/  real/  fake/    <- from glide
    test/
      glide/ real/ fake/  <- held-out glide images
      biggan/real/ fake/  <- cross-generator test

Subset design:
  glide  -> train 4000+4000 / val 1000+1000 / test 500+500
  biggan -> test  500+500
  Total  ~ 11 500 images
"""

import os
import random
import subprocess
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
SEED   = 42
HF_ENDPOINT = "https://hf-mirror.com"
HF_REPO     = "shimei123/Genimage"

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = PROJECT_ROOT / "data"
RAW_DIR      = DATA_ROOT / "raw" / "shimei"

TRAIN_PER_SIDE = 4000
VAL_PER_SIDE   = 1000
TEST_PER_SIDE  =  500

os.environ.setdefault("NO_PROXY", "*")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
random.seed(SEED)


# ── helpers ───────────────────────────────────────────────────────────────────

def download_file(filename: str) -> Path:
    dest = RAW_DIR / filename
    if dest.exists():
        print(f"  [skip] {filename} already downloaded")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    print(f"  Downloading {filename} …")
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=filename,
        local_dir=str(RAW_DIR),
    )
    return Path(path)


def list_zip_entries(zip_path: Path, folder: str) -> list[str]:
    """Return entry names inside zip that are under the given folder."""
    result = subprocess.run(
        ["unzip", "-l", str(zip_path)],
        capture_output=True, text=True
    )
    entries = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            name = parts[-1]
            if folder in name and not name.endswith("/"):
                entries.append(name)
    return entries


def extract_sample(zip_path: Path, folder: str, dest_dir: Path, n: int) -> int:
    """Sample n files from a zip folder and extract them flat to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    entries = list_zip_entries(zip_path, folder)
    chosen  = random.sample(entries, min(n, len(entries)))

    CHUNK = 200
    for i in range(0, len(chosen), CHUNK):
        batch = chosen[i:i+CHUNK]
        subprocess.run(
            ["unzip", "-j", "-o", str(zip_path)] + batch + ["-d", str(dest_dir)],
            capture_output=True,
        )
    rel = dest_dir.relative_to(PROJECT_ROOT)
    print(f"  {rel}: {len(chosen)} images")
    return len(chosen)


# ── per-generator steps ───────────────────────────────────────────────────────

def prepare_glide():
    print("\n=== Generator: glide (train / val / test) ===")
    zip_path = download_file("glide.zip")

    real_entries = list_zip_entries(zip_path, "0_real")
    fake_entries = list_zip_entries(zip_path, "1_fake")
    print(f"  Found: {len(real_entries)} real, {len(fake_entries)} fake")

    # Shuffle once, then slice to get non-overlapping train/val/test
    random.shuffle(real_entries)
    random.shuffle(fake_entries)

    tr_real = real_entries[:TRAIN_PER_SIDE]
    val_real = real_entries[TRAIN_PER_SIDE:TRAIN_PER_SIDE + VAL_PER_SIDE]
    te_real  = real_entries[TRAIN_PER_SIDE + VAL_PER_SIDE:
                            TRAIN_PER_SIDE + VAL_PER_SIDE + TEST_PER_SIDE]

    tr_fake  = fake_entries[:TRAIN_PER_SIDE]
    val_fake = fake_entries[TRAIN_PER_SIDE:TRAIN_PER_SIDE + VAL_PER_SIDE]
    te_fake  = fake_entries[TRAIN_PER_SIDE + VAL_PER_SIDE:
                            TRAIN_PER_SIDE + VAL_PER_SIDE + TEST_PER_SIDE]

    def batch_extract(entries, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        CHUNK = 200
        for i in range(0, len(entries), CHUNK):
            batch = entries[i:i+CHUNK]
            subprocess.run(
                ["unzip", "-j", "-o", str(zip_path)] + batch + ["-d", str(dest_dir)],
                capture_output=True,
            )
        print(f"  {dest_dir.relative_to(PROJECT_ROOT)}: {len(entries)} images")

    batch_extract(tr_real,  DATA_ROOT / "train" / "real")
    batch_extract(tr_fake,  DATA_ROOT / "train" / "fake")
    batch_extract(val_real, DATA_ROOT / "val"   / "real")
    batch_extract(val_fake, DATA_ROOT / "val"   / "fake")
    batch_extract(te_real,  DATA_ROOT / "test"  / "glide" / "real")
    batch_extract(te_fake,  DATA_ROOT / "test"  / "glide" / "fake")


def prepare_biggan():
    print("\n=== Generator: biggan (cross-gen test only) ===")
    zip_path = download_file("BigGAN.zip")
    extract_sample(zip_path, "0_real", DATA_ROOT / "test" / "biggan" / "real", TEST_PER_SIDE)
    extract_sample(zip_path, "1_fake", DATA_ROOT / "test" / "biggan" / "fake", TEST_PER_SIDE)


# ── stats ─────────────────────────────────────────────────────────────────────

def print_stats():
    print("\n=== Dataset statistics ===")
    total = 0
    for split in ("train", "val"):
        for label in ("real", "fake"):
            d = DATA_ROOT / split / label
            n = len(list(d.glob("*"))) if d.exists() else 0
            total += n
            print(f"  {split:5s}/{label:4s}: {n:6d}")
    for gen in ("glide", "biggan"):
        for label in ("real", "fake"):
            d = DATA_ROOT / "test" / gen / label
            n = len(list(d.glob("*"))) if d.exists() else 0
            total += n
            print(f"  test/{gen:6s}/{label:4s}: {n:6d}")
    print(f"  {'TOTAL':>17s}: {total:6d}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_glide()
    prepare_biggan()
    print_stats()
    print("\nDone. Data ready under ./data/")


if __name__ == "__main__":
    main()
