# Course resources and provenance

The requested website pages and Colab URL were attempted directly. The environment egress proxy returned HTTP 403 for `isom-835.vercel.app` and `colab.research.google.com`. Domain additions were saved to the environment draft; saving does not change live networking. The teaching sources were therefore read through the public HTTPS GitHub repository instead:

- Repository: https://github.com/harslan/isom-835
- Read source commit: `a3dd676287524f6fbcf7c7db43a7f31c54dd1047`
- `/project/`: `src/pages/project.astro` (local extracted copy `project_source.txt`).
- `/datasets/`: `src/pages/datasets.astro` (`datasets_source.txt`).
- `/cheatsheet/`: `src/pages/cheatsheet.astro` (`cheatsheet_source.txt`).
- `/sessions/session-03/`, `09`, `11`: `src/content/sessions/session-{03,09,11}.json` (local copies).
- Requested Colab notebook: `public/session-11/ISOM835_Session11_Forecasting.ipynb`; the identical raw GitHub path was accessible (local `session11.ipynb`).

The sources emphasize prediction framing, pipeline preprocessing within training, time-aware CV, MAE versus a naive baseline, genuine SHAP explanations, uncertainty, a model card, and a business decision with stated money assumptions. The project source permits disclosed AI assistance and requires students to understand and defend the work.

The user's stricter task requirements govern this implementation. In particular, the teaching notebook's synthetic fallback, interpolation of missing targets, actual target-hour weather, random-split comparison, and inclusive split slicing are not copied. This project uses only supplied real data, retains missing targets, excludes future observed weather, uses nonoverlapping chronological periods, and treats conformal coverage as uncertain under temporal shift. Original features include lag_48, rolling max, observed-history count, and a calendar commute interaction. The raw `season` code is not used as a model feature; sliced seasons are explicitly meteorological.

The course project page lists final notebooks, memo and repo link due December 14, 2026 at 5 PM. Session 11's notebook contains a conflicting December 13 note; confirm the applicable deadline in Canvas. Nothing has been published or submitted.
