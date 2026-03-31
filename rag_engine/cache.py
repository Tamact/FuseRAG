from typing import Any, Dict


def cache_put(cache: Dict[Any, Any], key: Any, value: Any, max_size: int) -> None:
    if max_size <= 0:
        return
    if key in cache:
        cache[key] = value
        return
    cache[key] = value
    if len(cache) > max_size:
        cache.pop(next(iter(cache)))

