# FGspectra

[![Latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://simonsobs.github.io/fgspectra/)
[![Build Status](https://travis-ci.com/simonsobs/fgspectra.svg?branch=master)](https://travis-ci.com/simonsobs/fgspectra)
[![codecov](https://codecov.io/gh/simonsobs/fgspectra/branch/master/graph/badge.svg)](https://codecov.io/gh/simonsobs/fgspectra)

Library for the evaluation of the SEDs and cross-spectra of astrophysical components.

Main modules are
* `frequency.py`: evaluation of astrophysical SEDs
* `power.py`: evaluation of anugular power spectra
* `cross.py`: evaluation of frequency cross-spectra

To get started, have a look at the notebooks in `fgspectra/notebook`

## Contributing
Current main contributors are Zack Li, Max Abitbol and Davide Poletti. Feel free to join: contributors are welcome!

We try to
* develop in short-lived branches that are merged into master
* write [numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) and PEP8 compliant code.

## Material
We'll eventually pull in code/reference material from
* [fgbuster](https://github.com/fgbuster/fgbuster)
* [BeFore](https://github.com/damonge/BFoRe_py)
* [tile-c](https://github.com/ACTCollaboration/tile-c)
* [Erminia/Jo's multifrequency likelihood on Lambda](https://lambda.gsfc.nasa.gov/product/act/act_fulllikelihood_get.cfm)

## Dependencies
* Python > 3
* numpy / scipy

## Installing
Since we're still putting this together, just install in developer mode for now.

```
pip install -e .
```

## Testing
Run `pytest` in the `fgspectra` directory. (work in progress)
