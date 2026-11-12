# Bike Sharing Predictive Analysis — source data

`hour.csv` is the original UCI Bike Sharing hourly dataset used in this project. It contains **17,379 observed hourly Capital Bikeshare records** from January 1, 2011 through December 31, 2012. The unmodified file has 17 columns. The original daily aggregate is unnecessary for this hourly workflow and is not included.

- Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Original archive: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- Hourly-file SHA-256: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`

The loader in [`bike_sharing.py`](../bike_sharing.py) verifies this checksum before modeling. A changed local or downloaded source must be investigated; it is not silently substituted or assigned a new expected checksum.

`cnt` is total completed rentals in the hour. `casual` and `registered` are its components and are excluded from predictors. The date and hour construct the timeline. Calendar information is assumed available before issuance. Observed temperature, apparent temperature, humidity, wind speed and weather situation are excluded from forecast-time inputs; observed weather may describe errors after the forecasts are made.

A complete hourly index contains 17,544 slots, exposing 165 missing observations. Missing targets remain missing. No synthetic rentals, future-aware target interpolation, station inventories or unmet-demand measurements are supplied by these data.

Citation: Fanaee-T, Hadi, and João Gama (2013), “Event labeling combining ensemble detectors and background knowledge,” *Progress in Artificial Intelligence*. DOI: https://doi.org/10.1007/s13748-013-0040-3.

The [UCI dataset page](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) lists the data under **Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))**, checked during the November 11, 2026 review. Retain the attribution and identify any changes when reusing the data. The tracked `hour.csv` remains unmodified. This data license is separate from Gurobi’s terms and from the project’s source code, for which no license has been selected.
