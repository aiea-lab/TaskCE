# Neuron formula viewer

Build a standalone HTML viewer from search result CSVs. Run commands from the repository root with the project Python environment.

Build the default SNLI viewer:

```sh
python -m theoretical.compositional_explanations.formula_ast.viewer
```

The command writes `data/formula_ast_viewer.html`. Open that file directly in a desktop browser. It contains its data, styles, and JavaScript, so it needs no server or network connection.

## Build from CSV files

Pass one CSV and an output path to build a single-layer viewer:

```sh
python -m theoretical.compositional_explanations.formula_ast.viewer --csv path/to/results.csv --output data/formula_ast_viewer.html
```

Pass multiple CSV paths after `--csv` to bundle layers. A parent directory named `model.*` supplies each layer label. Other CSVs use their filename without the extension. Labels must be unique.

```sh
python -m theoretical.compositional_explanations.formula_ast.viewer --csv path/to/model.layer1/results.csv path/to/model.layer2/results.csv --output data/formula_ast_viewer.html
```

Each CSV needs `neuron`, `formula`, and `iou` columns. Neuron IDs must be unique, nonnegative integers. IoU values must be between 0 and 1. The viewer converts formulas with `analysis.parse_formula` and preserves the resulting Z3 tree without simplification. Each bundled dataset records its resolved source path and the SHA-256 hash of the CSV bytes used to build it.

For this local eight-layer snapshot, rebuild the viewer in PowerShell:

```powershell
$paths = (Get-Content data/viewer_upgrade/sources.json -Raw | ConvertFrom-Json).local
.venv/Scripts/python.exe -m theoretical.compositional_explanations.formula_ast.viewer --csv $paths --output data/formula_ast_viewer.html
```

`sources.json` lists this snapshot's layers. For other inputs, pass the CSV paths directly as shown above.

## Browse and verify

Choose a layer from **Layer**. Previous and next follow that layer's CSV row order. The neuron jump and other-true-neuron list use that layer's records. Layer changes preserve the neuron ID when it exists in the new layer, the shared concept assignment, and the tree size preference. Each layer shows its own source path and CSV hash.

Concept clicks toggle a shared Boolean assignment. Unlisted concepts are false. Operator clicks show their calculation. The other-true-neuron list evaluates formulas under the same assignment and excludes the displayed neuron. These are formula outputs, not measured model activations or coactivation frequencies across a dataset.

`LOW_ACTS_PRUNED` rows remain in navigation and the neuron jump. They show **No formula** and **Not evaluated**. The other-true-neuron count excludes pruned rows and reports how many rows were pruned.

Verify that the generated HTML records, ASTs, source paths, and hashes match the CSV files:

```powershell
.venv/Scripts/python.exe -m theoretical.compositional_explanations.formula_ast.verify_viewer --html data/formula_ast_viewer.html --csv $paths
```

Pass the same CSV paths, in the same order, that you used to build the viewer.

`viewer.py` loads the CSVs and bundles the page. `viewer_assets/index.html`, `viewer.css`, and `viewer.js` define the interface. Edit those sources and rebuild the HTML. Generated output belongs in the ignored `data/` directory.
