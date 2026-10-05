"""Inject the public browser token into the deployment artifact, never Git."""
import os
from pathlib import Path
import re
import shutil

token = os.environ.get('MAPBOX_PUBLIC_TOKEN', '')
if not re.fullmatch(r'pk\.[A-Za-z0-9._-]+', token):
    raise SystemExit('Set MAPBOX_PUBLIC_TOKEN to a public (pk.) Mapbox token.')

root = Path(__file__).resolve().parents[1]
output = root / '_site'
if output.exists():
    raise SystemExit('_site already exists; use a clean checkout.')
shutil.copytree(root, output, ignore=shutil.ignore_patterns('.git', '.github', '_site'))
reports = output / 'vpts-extrapolation/analysis/radar-pairs'
expected = {p.name for p in reports.glob('*-examples.html')}
expected |= {name.replace('-examples.html', '-full-report.html') for name in list(expected)}
count = 0
for path in reports.glob('*.html'):
    html = path.read_text()
    if '__MAPBOX_PUBLIC_TOKEN__' in html:
        path.write_text(html.replace('__MAPBOX_PUBLIC_TOKEN__', token))
        count += 1
if not expected or count != len(expected):
    raise SystemExit(f'Expected {len(expected)} embedded map reports, found {count}.')
print(f'Prepared {count} map reports and the remaining static site.')
