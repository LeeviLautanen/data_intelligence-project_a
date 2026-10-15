## Data Intelligence - Project A

This is the repository for the STEWARD-AW pipeline required by project A. The pipeline predicts whether a candidate antibiotic will be effective or not, using the Antibiotic Resistance Microbiology Dataset (ARMD). The pipeline builds a leakage-safe analytical dataset (nine-antibiotic cohort) and compares logistic regression with LightGBM. The feature harmonization matrix and data dictionary can be found as excel files in the repository. 

To reproduce the results in the example_output directory, follow these steps:

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

5. View results in the created output folder. There should be two CSVs and two images. One of the CSVs is the processed data and one is the results of the prediction. The two images are calibration plots, one for the test data and one for the validation data. You can see example results in the folder "example_output".
