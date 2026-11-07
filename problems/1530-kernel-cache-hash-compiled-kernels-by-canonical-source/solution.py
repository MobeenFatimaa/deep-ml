import hashlib
import re

def canonical(src, name):
    """
    Replaces the specific function name with the placeholder 'K' 
    so that two kernels differing only in name compare equal.
    Uses regex to ensure only the whole function name is replaced.
    """
    # Use word boundaries around the function name to avoid partial substring matches
    pattern = r'\b' + re.escape(name) + r'\b'
    return re.sub(pattern, 'K', src)


class KernelCache:
    def __init__(self):
        self.compiles = 0
        self.hits = 0
        self.sources = {}

    def compile(self, src, name):
        """
        Computes the SHA-1 hash of the canonical source.
        Increments compiles if new, or hits if already cached.
        Returns the cache key (hex digest).
        """
        canon_src = canonical(src, name)
        key = hashlib.sha1(canon_src.encode('utf-8')).hexdigest()
        
        if key in self.sources:
            self.hits += 1
        else:
            self.compiles += 1
            self.sources[key] = canon_src
            
        return key


def dedupe_program(kernels):
    """
    Passes each kernel through a KernelCache.
    Returns a tuple of (unique_sources, calls), where unique_sources 
    is the number of compiles and calls is a list of (key, bufs) pairs.
    """
    cache = KernelCache()
    calls = []
    
    for kernel in kernels:
        key = cache.compile(kernel['src'], kernel['name'])
        calls.append((key, kernel['bufs']))
        
    return cache.compiles, calls


def compile_savings(kernels, compile_ms):
    """
    Calculates the total time saved in milliseconds due to kernel caching.
    """
    if not kernels:
        return 0.0
    
    cache = KernelCache()
    for kernel in kernels:
        cache.compile(kernel['src'], kernel['name'])
        
    unique_sources = cache.compiles
    return (len(kernels) - unique_sources) * compile_ms