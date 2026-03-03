# ml2
General Machine Learning ML functions.

## Environment and Package Management
We use `uv` to manage environments and packages.
```
brew install uv
```

### Create virtual environment
Example venv `ml`:
```
uv venv ml
source ml/bin/activate
```

#### To delete venv
```
deactivate
rm -rf ml
```

### Package Requirements
We use `pyproject.toml` to manage package dependencies. To read `pyproject.toml`, resolve
all dependencies, and install them:
```
uv pip install -e .
```
- `-e`: Editable mode. This means that any changes made to code in the src directory
will be immediately available without needing to reinstall
- `.[dev]`: This tells uv to install the project in the current directory (.) along
with the extra dependencies defined in the `[dev]` group

If there are installation issues with packages, trying installing without cached
package versions:
```
uv pip install -e . --no-cache
```
