"""Use the actual local UCI files, or download them with normal TLS verification."""
from pathlib import Path
import hashlib
import io
import zipfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
EXPECTED_HOUR='e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f'

def main():
    dest=ROOT/'data';dest.mkdir(exist_ok=True)
    hour=dest/'hour.csv'
    if not hour.exists():
        with urllib.request.urlopen(URL,timeout=120) as response:
            archive=zipfile.ZipFile(io.BytesIO(response.read()))
        # Explicit member names avoid extracting arbitrary paths.
        payload=archive.read('hour.csv')
        if hashlib.sha256(payload).hexdigest()!=EXPECTED_HOUR:
            raise ValueError('Source checksum differs from audited uploaded hour.csv; inspect source before use')
        hour.write_bytes(payload)
        for source,target in [('day.csv','day.csv'),('Readme.txt','UCI_Readme.txt')]:
            if not (dest/target).exists():(dest/target).write_bytes(archive.read(source))
    actual=hashlib.sha256(hour.read_bytes()).hexdigest()
    if actual!=EXPECTED_HOUR:raise ValueError('Local hour.csv does not match audited dataset')
    print(f'Verified {hour.name}: SHA256 {actual}')

if __name__=='__main__':main()
