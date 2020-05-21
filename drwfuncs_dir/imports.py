"""File import all necessary modules."""
from dependency import install_dependencies

# function check whether some dependencies are missing
try:
    install_dependencies()
except KeyboardInterrupt:
    print()
    exit()
# import necessary modules
import os
import sys
import numpy as np
from shut_yard_alg import Function
from optparse import OptionParser
