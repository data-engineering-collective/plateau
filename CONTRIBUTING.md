## How to contribute

### Bugs

If you've found a bug, please report it to the issue tracker and

- describe the bug you encountered and what the expected behavior should be,
- provide a [mcve](https://stackoverflow.com/help/mcve) if possible, and
- be as explicit about your environment as possible, e.g. provide a `pip freeze` / `conda list`.

### Development

#### Installation using pip

To get started, set up a a new environment using [pixi](https://pixi.sh) and install all requirements:

```bash
pixi run pre-commit-install
pixi run postinstall
pixi run test
```

#### Running tests

We're using [pytest](https://pytest.org) as a testing framework and make heavy use of
`fixtures` and `parametrization`.

To run the tests simply run

```bash
pixi run test
```

#### Running benchmarks

For performance critical code paths we have [asv](https://asv.readthedocs.io/en/latest/) benchmarks in place in the subfolder `asv_bench`.
To run the benchmarks a single time and receive immediate feedback run

```bash
pixi run -e benchmark postinstall
pixi run -e benchmark asv run --python=same --show-stderr --config ./asv_bench/asv.conf.json
```

#### Building documentation

```bash
pixi run -e docs postinstall
pixi run docs
```

#### Code style

To ensure a consistent code style across the code base we're using `prettier` and `ruff` for formatting and linting.

We have [pre-commit](https://pre-commit.com) hooks for all of these tools which take care of formatting
and checking the code.

If you prefer to perform manual formatting and linting, you can run the necessary
toolchain like this

```bash
pixi run pre-commit-run
```
