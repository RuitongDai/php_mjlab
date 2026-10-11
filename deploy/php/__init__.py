"""Deployment entry points for the PHP student."""

import os

os.environ.setdefault("PHP_ONLY", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/php-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/php-cache")
