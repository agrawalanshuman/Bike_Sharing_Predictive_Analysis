"""Build business deliverables after executing the advanced/final notebook."""
from pathlib import Path
import sys
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bike-mpl")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.documents import generate_pdfs
generate_pdfs()
print('Generated TeamName_dryrun.pdf and TeamName_memo.pdf from results/.')
