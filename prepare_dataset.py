import io
import ssl
import zipfile
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences"
]

NUMERIC = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G3"
]


def find_student_csv(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            if name.rsplit("/", 1)[-1].lower() == "student-mat.csv":
                return archive.read(name)

        for name in archive.namelist():
            if name.lower().endswith(".zip"):
                try:
                    result = find_student_csv(archive.read(name))
                    if result is not None:
                        return result
                except zipfile.BadZipFile:
                    continue

    return None


def main():
    print("Downloading the official UCI Student Performance dataset...")

    request = urllib.request.Request(
        URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    context = ssl.create_default_context(cafile=certifi.where())

    with urllib.request.urlopen(
        request, timeout=60, context=context
    ) as response:
        archive_bytes = response.read()

    csv_bytes = find_student_csv(archive_bytes)

    if csv_bytes is None:
        raise FileNotFoundError(
            "Could not find student-mat.csv inside the downloaded ZIP files."
        )

    df = pd.read_csv(io.BytesIO(csv_bytes), sep=";")
    print(f"Original dataset: {len(df)} rows")

    missing_columns = set(FEATURES + ["G3"]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset columns missing: {sorted(missing_columns)}")

    # Exclude G1 and G2 to avoid using earlier grades as prediction inputs.
    df = df[FEATURES + ["G3"]].copy()

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)

    before = len(df)
    df = df.dropna(subset=FEATURES + ["G3"])
    missing_rows_removed = before - len(df)

    df = df[df["G3"].between(0, 20)]
    df = df.reset_index(drop=True)

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "student-mat.csv"
    df.to_csv(output_path, index=False, sep=";")

    report = f"""UCI Student Performance Dataset - Cleaning Report

Source: UCI Machine Learning Repository
Dataset: Mathematics student performance
Original rows: {len(pd.read_csv(io.BytesIO(csv_bytes), sep=";"))}
Duplicate rows removed: {duplicates_removed}
Rows with missing required values removed: {missing_rows_removed}
Clean rows saved: {len(df)}
Input features: {len(FEATURES)}
Target column: G3
Excluded leakage columns: G1, G2
Target range: 0 to 20
Output file: {output_path}
"""

    (output_dir / "cleaning_report.txt").write_text(report)
    print(report)
    print("Dataset preparation completed successfully!")


if __name__ == "__main__":
    main()
