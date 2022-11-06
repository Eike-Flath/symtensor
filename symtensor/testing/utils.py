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

# Function for displaying source code in an IPython cell
# Especially useful for documenting a module in a notebook
import textwrap
import inspect
from typing import Union
from collections.abc import Callable
try:
    from IPython.display import display, HTML
except ModuleNotFoundError:
    # Fallbacks. If IPython is not loaded, most likely we are in a terminal.
    def display(s):
        return None
    class HTML:
        def __init__(self, s):
            self.content = str(s)
        def _repr_html_(self):
            return self.content
try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    from IPython.display import HTML, display
    pygments_loaded = True
except ModuleNotFoundError:
    pygments_loaded = False

class CodeStr:
    """
    Return a highlighted string of code for display in a notebook output cell.
    Todo: correct export to Latex
    """
    pygments_loaded=pygments_loaded

    def __init__(self, code):
        """
        Parameters
        ----------
        code: str
        """
        self.code = code
        if self.pygments_loaded:
            display(HTML("""
            <style>
            {pygments_css}
            </style>
            """.format(pygments_css=HtmlFormatter().get_style_defs('.highlight'))))

    def _repr_html_(self):
        if self.pygments_loaded:
            pythonlexer = PythonLexer(encoding='chardet')
            htmlformatter = HtmlFormatter()
            return HTML(data=highlight(self.code, pythonlexer, htmlformatter)) \
                ._repr_html_()
        else:
            return HTML(data="<pre>" + self.code + "</pre>")._repr_html_()

    def __str__(self):
        return str(self.code)

    def replace(self, old, new, count=-1):
        """Call `replace` on the code string and update with the result."""
        self.code = self.code.replace(old, new, count)
        return self

def Code(*obj, sep=''):
    """
    Extract object source code with `inspect.getsource` and return a string
    with syntax highlighting suitable for a notebook output cell.

    Parameters
    ----------
    *obj: objects for which we want to print the source code
    sep:  (keyword only) Optional separator, if multiple objects are given.
          This is appended to the already present newline.
    """
    src = sep.join([inspect.getsource(s) for s in obj])
    return CodeStr(textwrap.dedent(src).strip())  # .strip() removes final newline

class NBTestRunner:
    """Display and run standard pytests in a notebook."""
    def __init__(self, API_test_class, symtensor_cls, display: bool=False):
        """
        :param:API_test_class: A class containing pytest tests.
        :param:symtensor_cls: The `SymmetricTensor` subclass to which the tests apply
        :param:display: Whether to also display the test source code. Only applies when executed with IPython
        """
        if isinstance(API_test_class, type):  # Was passed a type instead of an instance
            API_test_class = API_test_class()
        self.API = API_test_class
        self.SymTensor = symtensor_cls
        self.display = display

    def __call__(self, test: Union[str, Callable]):
        """
        :param: The test to run; must be defined within `self.API`.
                May optionally be a string, in which case the corresponding test
                is retrieved.
        """
        if self.display:
            if isinstance(test, str):
                test = getattr(self.API, test)
            display( Code(test).replace("SymTensor", self.SymTensor.__name__) )
        # Workaround to emulate effect of fixtures
        args = []
        for argname in inspect.signature(test).parameters:
            if argname == "self":
                args.append(self.API)
            elif argname == "SymTensor":
                args.append(self.SymTensor)
            elif argname == "test_tensor":
                args.append(next(iter(self.API.get_test_tensors(self.SymTensor))))
            else:
                raise ValueError("NBTestRunner can only deal with hard-coded "
                                 f"fixtures. '{argname}' is not a recognized fixture.")
        test(*args)
