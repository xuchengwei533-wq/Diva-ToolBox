# ThreePlotting Scripts

This directory contains scripts for generating 3D scatter plots of extracted features.

## Scripts

1.  **ThreePlot_A1.py**: Generates 3D scatter plots for files with suffix "A" and "1". Outputs to `A1/`.
2.  **ThreePlot_B1.py**: Generates 3D scatter plots for files with suffix "B" and "1". Outputs to `B1/`.
3.  **ThreePlot_ALL.py**: Generates 3D scatter plots for all files. Outputs to `ALL/`.
4.  **ThreePlot_A1_B1_ALL.py**: Generates a combined figure with A1, B1, and ALL plots side-by-side. Outputs to `A1_B1_ALL/`.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the scripts using Python:

```bash
python ThreePlot_A1.py
python ThreePlot_B1.py
python ThreePlot_ALL.py
python ThreePlot_A1_B1_ALL.py
```

## Output

The plots will be saved in the corresponding subdirectories (`A1`, `B1`, `ALL`, `A1_B1_ALL`).
Each plot shows the relationship between three features (X, Y, Z), color-coded by the score (1, 3, 5).
