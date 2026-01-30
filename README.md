# RDM Kenya

RDM Kenya provides a Robust Decision Making (RDM) workflow and analysis tools for
running OSeMOSYS scenarios and post-processing results. It bundles the Python
package under `src/rdm_kenya` and the workflow assets required to execute the
scenario pipeline.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable installs of the local package:

```bash
pip install -e .
```

## Basic usage

The main entry point runs the RDM workflow using the configuration inputs under
`data/inputs` and the scenario files in `workflow/scenarios`:

```bash
python RUN_RDM.py
```

You can also import the package modules directly, for example:

```python
from rdm_kenya.experiment import run_rdm_workflow

run_rdm_workflow()
```

## Data layout

Key directories in this repository include:

- `data/inputs/`: CSV configuration inputs consumed by the workflow
  (`Setup.csv`, `To_Print.csv`, `Uncertainty_Table.csv`, etc.).
- `data/intermediate/`: Placeholder for intermediate outputs generated during
  processing.
- `workflow/scenarios/`: OSeMOSYS scenario text files used as inputs.
- `workflow/experiments/`: Generated executables and future datasets.
- `workflow/miscellaneous/`: Supporting model structure assets such as
  `OSeMOSYS_Structure.xlsx`.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for
full details.

## Citation

If you use this repository in academic work, please cite it as described in
[`CITATION.cff`](CITATION.cff).
