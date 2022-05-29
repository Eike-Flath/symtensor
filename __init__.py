# Placeholder module until we move symtensor to its own package
#
# The nesting of code files under a second *symtensor* directory is in
# preparation of making it its own package, but until that happens,
# the repeated appearance of 'symtensor in imports:
#
#     from statGLOW.stats.symtensor.symtensor.symtensor
#
# is confusing and unnecessary.

from .symtensor import *
