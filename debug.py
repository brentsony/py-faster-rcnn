from __future__ import print_function
import sys


def debug (*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

