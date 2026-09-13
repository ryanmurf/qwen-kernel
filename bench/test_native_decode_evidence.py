import copy
import json
from pathlib import Path
import unittest

from audit_native_decode_loads import validate_gate, validate_resources
from test_native_ktranspose import telemetry


class DecodeEvidenceTests(unittest.TestCase):
    def test_existing_exact_gate_and_corruption(self):
        root=Path(__file__).parent
        reference=[json.loads(line) for line in (root/'results-halo-last-head-full-gate.jsonl').read_text().splitlines()]
        rows=[json.loads(line) for line in (root/'results-halo-decode-loads/gate.jsonl').read_text().splitlines()]
        self.assertEqual(validate_gate(reference,rows)['rows'],664)
        broken=copy.deepcopy(rows)
        next(r for r in broken if r['type']=='row')['logit_sha256']='0'*64
        with self.assertRaises(ValueError): validate_gate(reference,broken)
        broken=copy.deepcopy(rows)
        broken.pop(-2)
        with self.assertRaises(ValueError): validate_gate(reference,broken)

    def test_resource_receipt_and_corruption(self):
        controller,rows=telemetry()
        self.assertEqual(validate_resources(rows,'test-unit',controller['final_unit'],False)['swap_peak_bytes'],0)
        for mutate in (lambda c,r:r[1]['memory'].update(MemAvailable=7*2**30),
                       lambda c,r:r[1].update(elapsed=19),
                       lambda c,r:r[-1]['properties'].update(SubState='stop-sigterm'),
                       lambda c,r:r[1]['events'].update(oom='1'),
                       lambda c,r:r[1]['temperatures'].update(temp1_input='94000'),
                       lambda c,r:c['final_unit'].update(MemorySwapPeak='4096'),
                       lambda c,r:r[1]['devices']['2'].update(**{'drm-engine-gfx':'1 ns'})):
            c,r=telemetry()
            mutate(c,r)
            with self.assertRaises(ValueError): validate_resources(r,'test-unit',c['final_unit'],False)


if __name__=='__main__': unittest.main()
