import os

import pandas as pd

# Standardized nine-antibiotic cohort
ANTIBIOTICS = [
    "Ampicillin",
    "Cefazolin",
    "Ceftriaxone",
    "Ciprofloxacin",
    "Gentamicin",
    "Levofloxacin",
    "Meropenem",
    "Nitrofurantoin",
    "Vancomycin",
]

# Columns used to identify a unique culture
IDS = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded"]

# Columns that define a unique row in the output dataset
KEY = IDS + ["organism", "antibiotic"]


def read(name, columns=None):
    # Return target columns from file (treat string "Null" as missing)
    return pd.read_csv(f"data/{name}.csv", usecols=columns, na_values=["Null"])


def load_labels():
    df = read("microbiology_cultures_cohort")

    # Filter the data to only our nine antibiotics
    df = df[df["antibiotic"].isin(ANTIBIOTICS)]
    print("rows for 9 antibiotics:", len(df))

    # Drop rows with conflicting results for susceptibility
    n_labels = df.groupby(KEY)["susceptibility"].transform("nunique")
    df = df[n_labels <= 1]
    print("rows after dropping conflicting keys:", len(df))

    # Drop susceptibility values that arent "susceptible" or "resistant"
    df = df[df["susceptibility"].isin(["Susceptible", "Resistant"])]

    # Convert susceptibility to a new column "label" with ones and zeroes
    df["label"] = (df["susceptibility"] == "Susceptible").astype(int)

    # Drop old columns that arent needed anymore
    df = df.drop(columns=["susceptibility", "was_positive"]).drop_duplicates()
    print("rows after keeping S/R only:", len(df))
    return df


def add_demographics(df):
    # Merge age and gender columns to input table
    demo = read("microbiology_cultures_demographics", IDS + ["age", "gender"])
    return df.merge(demo, on=IDS, how="left")


def add_ward(df):
    # Merge hospital ward columns to input table
    ward_cols = ["hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU"]
    ward = read("microbiology_cultures_ward_info", IDS + ward_cols)
    return df.merge(ward, on=IDS, how="left")


def add_prior_resistance(df):
    # Read resistance related data from table
    res = read(
        "microbiology_cultures_microbial_resistance",
        IDS + ["antibiotic", "resistant_time_to_culturetime"],
    )

    # Filter out irrelevant antibiotics and duplicates
    res = res[res["antibiotic"].isin(ANTIBIOTICS)]
    res = res[IDS + ["antibiotic"]].drop_duplicates()

    # Create column with 1's since every row left represents a resistance
    res["prior_resistant"] = 1

    # Merge the new column to the input table and set rows with no value to zero
    df = df.merge(res, on=IDS + ["antibiotic"], how="left")
    df["prior_resistant"] = df["prior_resistant"].fillna(0).astype(int)
    return df


df = load_labels()
n_rows = len(df)
df = add_demographics(df)
df = add_ward(df)
df = add_prior_resistance(df)

print("\nrows in final table:", len(df))
print("duplicate keys:", df.duplicated(KEY).sum())
print("\nmissing values per column:")
print(df.isna().sum())
print("\nsusceptible share by prior_resistant:")
print(df.groupby("prior_resistant")["label"].agg(["count", "mean"]).round(3))

os.makedirs("output", exist_ok=True)
df.to_csv("output/dataset.csv", index=False)
print("\nsaved output/dataset.csv")
