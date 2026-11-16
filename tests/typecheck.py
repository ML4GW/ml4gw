from contextlib import contextmanager

import jaxtyping
from beartype import BeartypeConf, beartype

# Accept ints where annotations specify floats
checker = beartype(conf=BeartypeConf(is_pep484_tower=True))


@contextmanager
def no_typecheck():
    """Turn off type checking to test ml4gw's own input validation."""
    jaxtyping.config.update("jaxtyping_disable", True)
    try:
        yield
    finally:
        jaxtyping.config.update("jaxtyping_disable", False)
