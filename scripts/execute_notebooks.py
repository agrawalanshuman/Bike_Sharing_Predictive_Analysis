"""Execute each real notebook from the repository root; stop on the first error."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
# Register an isolated, local kernel rather than relying on a machine's default Python.
kernels=ROOT/'.jupyter/kernels/bike835'
kernels.mkdir(parents=True,exist_ok=True)
import json
(kernels/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
 'display_name':'Bike835','language':'python','env':{'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MPLCONFIGDIR':'/tmp/bike-mpl'}}))
os.environ['JUPYTER_PATH']=str(ROOT/'.jupyter')
os.environ['JUPYTER_RUNTIME_DIR']='/tmp/bike-jupyter-runtime'
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
files=sys.argv[1:] or ['Anshuman_milestone2.ipynb','Anshuman_milestone3.ipynb','BikeSharing_Final.ipynb']
for filename in files:
    path=ROOT/filename
    nb=nbformat.read(path,as_version=4)
    print('Executing',filename,flush=True)
    try:
        NotebookClient(nb,timeout=600,kernel_name='bike835',resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        nbformat.write(nb,path)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    print('Executed successfully:',filename,len(code),'code cells',flush=True)
