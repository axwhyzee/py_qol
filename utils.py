from functools import wraps
from hashlib import md5
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional, Iterator
import re


logger = logging.getLogger()

CACHE_DIR = '.cache_disk'


def crawl_dir(root_path: Path, file_regex: str | None = None) -> Iterator[Path]:
    """Recursively search for files with optional regex starting froma root_path"""
    for path in root_path.iterdir():
        if path.is_dir():
            yield from crawl_dir(path, file_regex)
        else:
            if not file_regex:
                yield path
            elif re.search(file_regex, str(path)):
                yield path
                

def timer(
    func: Callable
) -> Callable:
    """Calculate decorated function duration"""

    @wraps(func)
    def inner(*args, **kwargs) -> Any:
        st = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f'({func.__name__}) finished in {(end-st):.4f}s')
        return res
    return inner


def cache_disk(
    *,
    save_fn: Optional[Callable] = None, 
    load_fn: Optional[Callable] = None
) -> Callable:
    """Persist decorated function outputs to disk, identified by 
    function name and stringified args.

    If data to save is composed of custom types, serialization into
    pickle files can be severely hampered, in which cases it could be 
    faster to provide a different method of saving and loading by 
    setting the save_fn and load_fn args.

    Args:
        save_fn (Optional[Callable]): Function to save results via
        load_fn (Optional[Callable]): Function to load results using

    Example::

        import xmltodict
        import zipfile
        import os
        from utils import cache_disk

        @cache_disk()
        def parse_zipped_xml(path: str) -> dict:
            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(".tmp")
            
            file_name = os.listdir(".tmp")[0]
            file_name = os.path.join(".tmp", file_name)
            return xmltodict.parse(file_name)
    """

    def _cache_disk(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not os.path.exists(CACHE_DIR):
                os.mkdir(CACHE_DIR)

            f_name = func.__name__
            call_hash = md5((str(args) + str(kwargs)).encode('utf-8')).hexdigest()
            cache_path = os.path.join(CACHE_DIR, f'{f_name}_{call_hash}')

            # cache exists, load
            if os.path.exists(cache_path):
                logger.info(f'Found existing cache for [{f_name}] at [{cache_path}]')
                
                if load_fn:
                    return load_fn(cache_path)
                with open(cache_path, 'rb') as fh:
                    return pickle.load(fh)

            # cache does not exist, calculate and cache
            res = func(*args, **kwargs)
            try:
                if save_fn:
                    save_fn(res, cache_path)
                else:
                    with open(cache_path, 'wb') as fh:
                        pickle.dump(res, fh)
            except Exception:
                # cleanup if save failed
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            finally:
                logger.info(f'Cached [{f_name}] results to [{cache_path}]')
            return res
        return wrapper
    return _cache_disk
