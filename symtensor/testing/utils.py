from contextlib import contextmanager
import pytest
@contextmanager
def does_not_warn(*args, **kwargs):
    """
    Inverse of a `pytest.warns` test: raises `Failed` if the warning was not
    emitted. Arguments are the same as for `pytest.warns`.
    """
    from _pytest.outcomes import Failed
    try:
        with pytest.warns(*args, **kwargs):
            yield
    except Failed:
        pass
    else:
        raise Failed("A warning was emitted.")  # TODO: Recover warning from pytest
