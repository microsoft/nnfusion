# NNFusion Python Tests

## Installation

```bash
cd path/to/nnfusion/test/python

python3 -m pip install -r requirements.txt
```

## Running tests

```bash
# A link is needed temporarily since installing locally (pip install -e .) is not yet supported.
ln -s ../../src/python/nnfusion nnfusion

pytest -v --forked
```
