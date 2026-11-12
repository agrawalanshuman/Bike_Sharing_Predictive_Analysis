"""Explicit notebook refresh; use a separate checkout to preserve saved outputs."""
from pathlib import Path
import json
import os
from .config import PROJECT_ROOT

def execute_notebooks(root=PROJECT_ROOT):
    """Execute each presentation of the analysis from the same project root."""
    root = Path(root)
    import sys
    import nbformat
    from nbclient import NotebookClient
    kernels = root/'.jupyter/kernels/bike-sharing'
    kernels.mkdir(parents=True,exist_ok=True)
    (kernels/'kernel.json').write_text(json.dumps({
        'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Bike Sharing','language':'python',
        'env':{'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MPLCONFIGDIR':'/tmp/bike-mpl'}}))
    os.environ['JUPYTER_PATH']=str(root/'.jupyter')
    os.environ['JUPYTER_RUNTIME_DIR']='/tmp/bike-jupyter-runtime'
    Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
    for filename in ['analysis.ipynb']:
        path=root/filename
        nb=nbformat.read(path,as_version=4)
        print('Executing',filename,flush=True)
        try:
            NotebookClient(nb,timeout=600,kernel_name='bike-sharing',resources={'metadata':{'path':str(root)}}).execute()
        finally:
            nbformat.write(nb,path)
        cells=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        print('Completed:',filename,len(cells),'code cells',flush=True)
