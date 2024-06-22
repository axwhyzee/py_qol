from pathlib import Path
from typing import Iterator
import re

def crawl_dir(root_path: Path, file_regex: str | None = None) -> Iterator[Path]:
    for path in root_path.iterdir():
        if path.is_dir():
            yield from crawl_dir(path, file_regex)
        else:
            if not file_regex:
                yield path
            elif re.search(file_regex, str(path)):
                yield path
