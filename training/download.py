"""Step 1 - Download the public NHANES files from the CDC website.

18 files (6 per survey cycle), about 95 MB in total, free and public domain. Already-downloaded
files are skipped, so running this twice does not download anything again.
"""
import time
import urllib.request

from .config import CYCLES, FILES, RAW_DIR, URL


def download_all(verbose: bool = True) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, (year, label) in CYCLES.items():
        for file in FILES:
            target = RAW_DIR / f"{file}_{suffix}.xpt"
            if target.exists() and target.stat().st_size > 10_000:
                continue
            url = URL.format(year=year, file=file, suffix=suffix)
            if verbose:
                print(f"     downloading {file} {label} ...")
            for attempt in range(3):
                try:
                    data = urllib.request.urlopen(url, timeout=120).read()
                    if data[:6] != b"HEADER":  # real XPT files start with "HEADER RECORD"
                        raise ValueError(f"{url} did not return an XPT file")
                    tmp = target.with_suffix(".part")
                    tmp.write_bytes(data)
                    tmp.replace(target)  # never leave a half-written file behind
                    break
                except Exception as err:  # retry politely, then give up
                    if attempt == 2:
                        raise
                    print(f"       retrying ({err})")
                    time.sleep(5)
            time.sleep(1)  # be gentle with the CDC server


if __name__ == "__main__":
    download_all()
