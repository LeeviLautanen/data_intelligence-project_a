## Data Intelligence - Project A

This is the repository for the STEWARD-AW pipeline required by project A. To run this pipeline you will need to follow the instructions below.

1. Download the following tables from the link below and place the data into a directory called "data".

- microbiology_cultures_cohort
- microbiology_cultures_demographics
- microbiology_cultures_ward_info
- microbiology_cultures_microbial_resistance

Link: https://datadryad.org/dataset/doi:10.5061/dryad.jq2bvq8kp

2. Create a python environment and install dependencies

```
py -m venv .venv
.venv/scripts/activate
pip install -r requirements.txt
```

3. Run the "build_dataset.py" script to create the dataset used for training, testing and validation

```
py ./build_dataset.py
```

4. Train and run the two models

```
py ./train.py
```

5. View results in the created output folder. There should be a csv with the dataset and an image showing a calibration plot.