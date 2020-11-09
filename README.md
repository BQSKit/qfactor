# Quantum Fast Circuit Optimizer (qFactor)

This optimizer can optimize the distance between a circuit, a sequence of
unitary gates, and a target unitary matrix. This optimizer uses an analytic
method based on the SVD operation.

## Installation

The best way to install this python package is with pip.

```
git clone git@github.com:edyounis/qfactor.git
cd qfactor
pip install .
```

## Usage

The qfactor package can be used to optimize a specified circuit.
You can specify your circuit with a list of Gate objects and pass them to
the optimize function. See [an example](https://github.com/edyounis/csvdopt/blob/master/examples/toffoli_synthesis.py).

