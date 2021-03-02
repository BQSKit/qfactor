# Quantum Fast Circuit Optimizer (Qfactor)

This optimizer can optimize the distance between a circuit, a sequence of
unitary gates, and a target unitary matrix. This optimizer uses an analytic
method based on the SVD operation.

## Installation

The best way to install this python package is with pip.

```
pip install qfactor
```

## Usage

The qfactor package can be used to optimize a specified circuit.
You can specify your circuit with a list of Gate objects and pass them to
the optimize function. See [an example](https://github.com/edyounis/qfactor/blob/master/examples/toffoli_synthesis.py).

## Copyright

Quantum Fast Circuit Optimizer (qFactor) Copyright (c) 2020, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals
from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

