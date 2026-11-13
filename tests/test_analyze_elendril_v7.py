"""Owner-intake engineering fixtures; not actual GPU experiment results."""
import json
import zipfile
import pytest
from tools.analyze_elendril_v7 import analyze
from tools.elendril_v7_export import export_results
from tools.elendril_cloud import fingerprint
from anra_v5.elendril.v7_protocol import PROTOCOL_SHA


def test_verified_intake_preserves_originals_and_does_not_claim_science(tmp_path):
    output=tmp_path/'output'; (output/'analysis').mkdir(parents=True)
    (output/'analysis'/'identity.json').write_text(json.dumps({'source':'a'*40,'protocol_sha256':PROTOCOL_SHA}))
    (output/'analysis'/'summary.json').write_text('{"fixture":"original owner bytes"}')
    exports=export_results(output,source='a'*40,protocol_sha=PROTOCOL_SHA,campaign={'status':'NO_GO'})
    path=output/'exports'/'ELENDRIL_V7_FULL.zip'; before=fingerprint(path)
    result=analyze(path,tmp_path/'archive','fixture')
    target=tmp_path/'archive'/'fixture'
    assert result['scientific_conclusion']=='CAUSE UNRESOLVED'
    assert result['independent_action_replay']['status']=='NOT_RUN'
    assert fingerprint(path)==before
    assert (target/'raw'/'analysis'/'summary.json').read_text()=='{"fixture":"original owner bytes"}'
    assert json.loads((target/'analysis'/'summary.json').read_bytes())['status']=='NOT_RUN'
    assert 'CAUSE UNRESOLVED' in (target/'analysis'/'scientific_report.md').read_text()
    with pytest.raises(FileExistsError): analyze(path,tmp_path/'archive','fixture')


def test_tampered_input_not_archived(tmp_path):
    path=tmp_path/'broken.zip'; path.write_bytes(b'not a zip')
    with pytest.raises(zipfile.BadZipFile): analyze(path,tmp_path/'archive','bad')
    assert not (tmp_path/'archive'/'bad').exists()
