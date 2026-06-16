import builtins

_debug_enabled = False


def enable_debug():
    global _debug_enabled
    _debug_enabled = True


def debug_print(*args, **kwargs):
    if _debug_enabled:
        builtins.print(*args, **kwargs)
