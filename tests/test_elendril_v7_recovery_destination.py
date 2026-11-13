"""Exercise actual recovery dispatch and copying; CUDA is mocked, no training."""
import json
import pytest
import torch
from tools import elendril_v7_worker as worker
from anra_v5.elendril.v7_protocol import PROTOCOL_SHA


def test_recovery_dispatch_leaves_copy_destination_new(tmp_path,monkeypatch):
    import anra_v5.elendril.v6_data as data
    import anra_v5.elendril.training as training
    import tools.elendril_action_cloud as cloud
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(data,'read_dataset',lambda _:object())
    monkeypatch.setattr(training,'runtime_identity',lambda _: {})
    monkeypatch.setattr(cloud,'session_identity',lambda:'engineered-test-session')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-fixture')
    original=tmp_path/'head2'; original.mkdir(); (original/'committed').write_bytes(b'preserved')
    challenge=tmp_path/'challenge.json'
    challenge.write_text(json.dumps({'source':'a'*40,'protocol_sha256':PROTOCOL_SHA,
        'gpu_uuid':'GPU-fixture','head2_path':str(original)}))
    destination=tmp_path/'new-recovery'
    class ReachedRestore(Exception): pass
    def restore(*args,**kwargs):
        assert (destination/'committed').read_bytes()==b'preserved'
        raise ReachedRestore()
    monkeypatch.setattr(worker,'create',restore)
    with pytest.raises(ReachedRestore):
        worker.run('recover',{'source':'a'*40,'timecap_seconds':1,'output':str(tmp_path/'analysis'),
            'checkpoint_root':str(destination),'job':{},'data_file':'fixture','challenge':str(challenge)})
