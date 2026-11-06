"""Download and organize the UTD-MHAD dataset (Inertial + RGB modalities).

Official page: https://personal.utdallas.edu/~kehtar/UTD-MHAD.html

Usage:
    python setup_data.py                    # download, extract, verify
    python setup_data.py --zip-dir ~/dl     # use already-downloaded zips
    python setup_data.py --verify-only      # just check file counts

Result:
    data/utd-mhad/Inertial/a{action}_s{subject}_t{trial}_inertial.mat
    data/utd-mhad/RGB/a{action}_s{subject}_t{trial}_color.avi
"""

import argparse
import logging
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("setup_data")

# Archive links from the official UTD-MHAD page. If they move, download the
# zips manually from the page above and pass --zip-dir.
BASE_URL = "https://personal.utdallas.edu/~kehtar/UTD-MAD"
MODALITIES = {
    # name: (zip filename, target dir, file suffix)
    "Inertial": ("Inertial.zip", config.INERTIAL_DIR, "_inertial.mat"),
    "RGB": ("RGB.zip", config.RGB_DIR, "_color.avi"),
}


def download(url: str, dest: Path, retries: int = 10, chunk: int = 1 << 20) -> None:
    """Download with HTTP Range resume; a partial `.part` file is continued."""
    log.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        total = int(r.headers["Content-Length"])
    with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for attempt in range(1, retries + 1):
            done = tmp.stat().st_size if tmp.exists() else 0
            bar.n = done
            bar.refresh()
            if done >= total:
                break
            req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab") as f:
                    if done and r.status != 206:
                        raise IOError("server ignored Range request; cannot resume")
                    while block := r.read(chunk):
                        f.write(block)
                        bar.update(len(block))
            except (OSError, urllib.error.URLError) as e:
                log.warning("Attempt %d/%d interrupted at %d bytes: %s",
                            attempt, retries, tmp.stat().st_size, e)
    if tmp.stat().st_size != total:
        raise IOError(f"incomplete download: {tmp.stat().st_size} of {total} bytes")
    tmp.replace(dest)


def extract_and_organize(zip_path: Path, target_dir: Path, suffix: str) -> int:
    """Extract only files matching `suffix` into a flat `target_dir`.

    The archives may nest files in subfolders, so we match by basename rather
    than relying on the internal layout.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and m.filename.endswith(suffix)
                   and not Path(m.filename).name.startswith("._")]
        for m in tqdm(members, desc=f"Extracting {zip_path.name}"):
            out = target_dir / Path(m.filename).name
            if out.exists() and out.stat().st_size == m.file_size:
                continue
            with zf.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    log.info("Extracted %d new files into %s", n, target_dir)
    return n


def _missing(target_dir: Path, suffix: str) -> list[str]:
    """Expected files for the selected actions that are absent from target_dir."""
    names = [f"a{a}_s{s}_t{t}{suffix}" for a in config.SELECTED_ACTIONS
             for s in config.ALL_SUBJECTS for t in config.TRIALS]
    return [n for n in names if not (target_dir / n).exists()]


def verify() -> bool:
    """Check that every selected (action, subject, trial) exists in both modalities."""
    expected = [(a, s, t) for a in config.SELECTED_ACTIONS
                for s in config.ALL_SUBJECTS for t in config.TRIALS]
    checks = {
        "Inertial": (config.INERTIAL_DIR, config.INERTIAL_FILE_FMT),
        "RGB": (config.RGB_DIR, config.RGB_FILE_FMT),
    }
    all_ok = True
    for name, (d, fmt) in checks.items():
        missing = [(a, s, t) for a, s, t in expected
                   if not (d / fmt.format(action=a, subject=s, trial=t)).exists()]
        found = len(expected) - len(missing)
        log.info("%s: %d / %d expected files present", name, found, len(expected))
        for a, s, t in missing:
            log.warning("  missing %s", fmt.format(action=a, subject=s, trial=t))
        if found == 0:
            all_ok = False
    return all_ok


def main() -> int:
    """Download, extract and verify the dataset; return a process exit code."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip-dir", type=Path, default=None,
                   help="Directory with pre-downloaded Inertial.zip / RGB.zip")
    p.add_argument("--keep-zips", action="store_true",
                   help="Keep downloaded zips in data/ instead of a temp dir")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    if not args.verify_only:
        if args.zip_dir:
            zip_dir = args.zip_dir.expanduser()
            tmp_ctx = None
        elif args.keep_zips:
            zip_dir = config.DATA_DIR.parent / "zips"
            zip_dir.mkdir(parents=True, exist_ok=True)
            tmp_ctx = None
        else:
            tmp_ctx = tempfile.TemporaryDirectory()
            zip_dir = Path(tmp_ctx.name)

        try:
            for name, (zip_name, target_dir, suffix) in MODALITIES.items():
                if not _missing(target_dir, suffix):
                    log.info("%s: all selected files already present in %s, skipping",
                             name, target_dir)
                    continue
                zip_path = zip_dir / zip_name
                if not zip_path.exists():
                    if args.zip_dir:
                        log.error("%s not found in %s", zip_name, zip_dir)
                        return 1
                    try:
                        download(f"{BASE_URL}/{zip_name}", zip_path)
                    except Exception as e:  # noqa: BLE001
                        log.error("Download of %s failed: %s", zip_name, e)
                        log.error("Download it manually from %s and rerun with "
                                  "--zip-dir <folder>",
                                  "https://personal.utdallas.edu/~kehtar/UTD-MHAD.html")
                        return 1
                try:
                    extract_and_organize(zip_path, target_dir, suffix)
                except zipfile.BadZipFile:
                    log.error("%s is not a valid zip (server may have returned "
                              "an HTML page). Download it manually.", zip_path)
                    return 1
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
