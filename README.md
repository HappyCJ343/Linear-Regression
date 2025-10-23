# Linear Regression Power Plant Experiment

This repository contains a compact implementation of the linear regression
experiment described in the "线性回归模型实验指导" hand-out. The project trains a
linear model to predict the net hourly electrical energy output (`PE`) of a
combined cycle power plant from four environmental measurements (`AT`, `V`,
`AP`, `RH`).

## Project layout

```
.
├── data/                     # Sample copy of the Folds5x2_pp dataset
│   └── Folds5x2_pp.csv
├── plots/                    # Generated automatically when you run the script
├── requirements.txt          # Python dependencies
├── src/
│   └── linear_regression.py  # Command line experiment script
└── README.md
```

The dataset bundled in `data/` contains the header and the first 20 samples from
the original UCI "Combined Cycle Power Plant" dataset so the workflow can be
executed offline. You can replace it with the full dataset if you have access to
it.

## Environment setup

The experiment only relies on a handful of standard scientific Python packages.
You can install them in a virtual environment of your choice:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The minimum supported Python version is 3.10, matching the versions required
> in the original lab instructions (Python 3.7 or newer also works for the
> dependencies used here).

## Running the experiment

Execute the script with Python to train the model, evaluate it on a random test
split and create the visualisation of predicted versus actual power output.

```bash
python -m src.linear_regression
```

By default the script uses the dataset located at `data/Folds5x2_pp.csv`, holds
25% of the rows for evaluation, prints the model coefficients and performance
metrics, and saves a scatter plot to `plots/actual_vs_predicted.png`.

Use `--help` to explore the available options:

```bash
python -m src.linear_regression --help
```

Important flags include:

- `--data`: provide an alternative path to the dataset CSV.
- `--test-size`: choose the fraction of the dataset reserved for evaluation.
- `--random-state`: make the train/test split reproducible.
- `--save-plot`: control where the scatter plot is written (directories are
  created automatically).
- `--show-plot`: open the plot interactively after saving it.

## Sample output

```
Model coefficients:
    AT: -0.4235
     V: -0.9308
    AP:  1.1172
    RH: -0.6056
Intercept:  475.7286

Evaluation metrics on the test set:
  Mean Squared Error (MSE): 48.7743
  Root Mean Squared Error (RMSE): 6.9853
  Mean Absolute Error (MAE): 5.8362
  R^2 Score: 0.4429

Scatter plot saved to: /path/to/repo/plots/actual_vs_predicted.png
```

The metrics and the resulting scatter plot will vary when you change the
random seed, test split size, or dataset.

## Visualisation

The generated scatter plot mirrors the one shown in the experiment manual: the
predicted power outputs are plotted against the measured values along with an
ideal `y = x` reference line to highlight the fit quality of the model.
