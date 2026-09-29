"""Fetch public datasets declared in configs/data.yaml.

Downloads only datasets with a resolvable URL and records their license.
Self-collected data is managed separately (see docs/DATASET.md).
"""
from __future__ import annotations

import argparse

from riverguard.config import load_config


def main() -> None:
    from riverguard.utils.console import utf8_console

    utf8_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for ds in cfg["datasets"].get("public", []):
        url = ds.get("url", "")
        if url.startswith("<"):
            print(f"[skip] {ds['name']}: URL chưa được điền.")
            continue
        print(f"[todo] download {ds['name']} from {url} (license: {ds.get('license')})")
        # TODO: implement actual download + checksum verification.


if __name__ == "__main__":
    main()
