# Source data

`hour.csv`, `day.csv`, and `UCI_Readme.txt` were provided by the user as files from the UCI Bike Sharing dataset. Hourly modeling uses only `hour.csv`; the daily aggregate is retained for reference and is not joined into forecast inputs.

Public source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Archive: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip

Citation: Fanaee-T, Hadi, and João Gama (2013), “Event labeling combining ensemble detectors and background knowledge,” Progress in Artificial Intelligence. DOI: https://doi.org/10.1007/s13748-013-0040-3

The UCI source documents completed Capital Bikeshare hourly/daily rentals and weather/calendar information from 2011–2012. The provided source readme is preserved, including its citation guidance. No new license is assigned to the source data or course references by this repository.

The audited hourly checksum is recorded in `results/summary.json` and enforced by `scripts/download_data.py`. The local uploaded source is used without downloading or inventing replacements. A later changed upstream archive triggers an inspection error rather than silent checksum replacement.
