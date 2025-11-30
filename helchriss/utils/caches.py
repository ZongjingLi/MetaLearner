import collections
import threading

def cached_property(fget):
    """A decorator that converts a function into a cached property. Similar to ``@property``, but the function result is cached. This function has threading lock support."""

    mutex = collections.defaultdict(threading.Lock)
    cache = dict()

    def impl(self):
        nonlocal impl
        id_ = id(self)
        with mutex[id_]:
            if id_ not in cache:
                cache[id_] = fget(self)
                return cache[id_]
            else:
                return cache[id_]

    return property(impl)
