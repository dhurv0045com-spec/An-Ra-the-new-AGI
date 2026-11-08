"""Keep owner payloads out of curated Git result folders."""
import json
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('version,run',[
    ('V7','c9981309a94847a4a97d8115c56d75c8'),
    ('V7.1','20c80f66e76e4b589aed2053afbddffc')])
def test_curated_results_retain_hashes_without_owner_payloads(version,run):
    folder=ROOT/'Experiment Result'/version
    files=[p for p in folder.rglob('*') if p.is_file()]
    assert files and all(p.suffix in ('.md','.json') for p in files)
    assert not (folder/run/'FINAL_STATUS.json').exists()
    provenance=json.loads((folder/run/'PROVENANCE.json').read_bytes())
    assert provenance['run_id']==run and len(provenance['source'])==40
    assert provenance['full_checkpoint_archive_independently_checked'] is False
    assert all(len(v['zip_sha256'])==64 and v['zip_bytes']>0 for v in provenance['exports'].values())
    for name in ('OWNER_ANALYSIS.zip','FINAL_STATUS.json','raw/receipt.json','plot.png','duplicate.csv'):
        check=subprocess.run(['git','check-ignore','--no-index','--quiet',str((folder/run/name).relative_to(ROOT))],cwd=ROOT)
        assert check.returncode==0,name
    assert subprocess.run(['git','check-ignore','--no-index','--quiet',str((folder/run/'PROVENANCE.json').relative_to(ROOT))],cwd=ROOT).returncode==1
