"""
Circuit SVD Optimizer

This package provides functions to optimize the unitaries in a circuit
with respect to some target unitary matrix.
"""

# Initialize Logging
import logging
_logger = logging.getLogger( "csvdopt" )
_logger.setLevel(logging.CRITICAL)
_handler = logging.StreamHandler()
_handler.setLevel( logging.DEBUG )
_fmt = "%(levelname)-8s | %(message)s"
_formatter = logging.Formatter( _fmt )
_handler.setFormatter( _formatter )
_logger.addHandler( _handler )

# Main API
from .gate import Gate
from .optimize import optimize

