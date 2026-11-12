"""Checksum-verified UCI loading; missing target hours are preserved."""
from pathlib import Path
import hashlib
import pandas as pd
from .config import PROJECT_ROOT, DATA_SHA256

def load_data(path=None):
    """Keep missing target hours as NaN, never interpolate rental observations."""
    path = Path(path or PROJECT_ROOT / 'data/hour.csv')
    expected = DATA_SHA256
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Hourly source checksum differs from the audited UCI dataset')
    raw = pd.read_csv(path)
    ts = pd.to_datetime(raw['dteday']) + pd.to_timedelta(raw['hr'], unit='h')
    if ts.duplicated().any():
        raise ValueError('Duplicate hourly timestamps')
    if raw['cnt'].isna().any() or not (raw['casual'] + raw['registered']).equals(raw['cnt']):
        raise ValueError('Invalid original target data')
    raw.index = pd.DatetimeIndex(ts, name='timestamp')
    return raw.sort_index(), raw.sort_index().asfreq('h')


def ensure_source_data(data_file=None, root=PROJECT_ROOT):
    """Use real supplied data or verified HTTPS source; never synthesize data."""
    import urllib.request
    import io
    import zipfile
    expected = DATA_SHA256
    dest = Path(root)/'data/hour.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = Path(data_file).expanduser().resolve() if data_file else PROJECT_ROOT/'data/hour.csv'
    if data_file is not None:
        if not local.is_file():
            raise FileNotFoundError('Explicit --data-file does not exist or is not a file')
        if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
            raise ValueError('Explicit --data-file differs from the audited UCI source')
    if not dest.exists():
        if local.is_file():
            payload = local.read_bytes()
        else:
            urls = [
                'https://raw.githubusercontent.com/agrawalanshuman/Bike_Sharing_Predictive_Analysis/6a75d26e7e408164a8672c6d3d30d76a3efbff32/data/hour.csv',
                'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
            ]
            errors = []
            for url in urls:
                try:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        payload = response.read()
                    if url.endswith('.zip'):
                        payload = zipfile.ZipFile(io.BytesIO(payload)).read('hour.csv')
                    if hashlib.sha256(payload).hexdigest() != expected:
                        raise ValueError('Dataset checksum mismatch')
                    break
                except Exception as exc:
                    errors.append(f'{url}: {type(exc).__name__}: {exc}')
            else:
                raise RuntimeError('Real data could not be loaded. Supply --data-file hour.csv.\n'+'\n'.join(errors))
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Supplied data differ from the audited UCI source')
        dest.write_bytes(payload)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
        raise ValueError('Existing output data differ from the audited UCI source')
    return dest
