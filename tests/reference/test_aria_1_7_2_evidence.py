"""Source-bound single-core acceptance for the v1.7.2 reference optimization."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2] / 'references/qualification/aria_1_7_2_search'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('name', ['baseline_p4', 'final_f12', 'final_c3', 'final_heldout'])
def test_reference_evidence_binds_numeric_protocol_and_vendor_results(name):
    project = ROOT / name
    manifest = read(project / 'manifest.json')
    record = read(project / 'qualification.json')
    protocol = read(project / 'rtl/protocol.json')
    assert record['manifest_sha256'] == protocol['manifest_sha256'] == digest(project / 'manifest.json')
    assert record['source_closure_sha256'] == manifest['source_closure_sha256'] == protocol['source_closure_sha256']
    assert record['rtl_cosimulation'] == protocol['status'] == 'passed'
    assert protocol['reset_aborts'] and protocol['reset_between_epochs']
    assert protocol['input_gap_cycles'] and protocol['output_stall_cycles']
    assert protocol['completed_samples'] >= manifest['verification']['corpora']['built_in']['sample_count']
    assert protocol['log_sha256'] == digest(project / 'rtl/simulation.txt')
    assert protocol['testbench_sha256'] == digest(project / 'rtl/testbench.sv')
    for filename, expected in protocol['vector_files'].items():
        assert digest(project / 'rtl' / filename) == expected
    for filename, expected in record['report_files'].items():
        assert digest(project / 'hls' / filename) == expected
    assert manifest['verification']['transformation_equivalence'] == 'passed'
    assert manifest['verification']['source_conversion_consistency'] == 'passed'
    assert manifest['verification']['stage_boundaries']['status'] == 'passed'
    if name.startswith('final'):
        assert manifest['schema_version'] == 8
        assert manifest['ravel']['release'] == '1.7.2'
        search = manifest['resolved_design']['optimization_search']
        assert search['target'] == {'ii_cycles': 85, 'status': 'predicted-met'}
        selected = next(c for c in search['candidates'] if c['id'] == search['selected_candidate'])
        assert selected['selectable'] and selected['arithmetic_schedule']['reuse_factor'] == 2
        assert all(s.get('arithmetic', {}).get('proof', {}).get('status') == 'proven'
                   for s in manifest['resolved_design']['stages'][:2])


def test_reference_meets_single_core_resource_and_routed_timing_acceptance():
    current = read(ROOT / 'final_f12/qualification.json')
    baseline = read(ROOT / 'baseline_p4/qualification.json')
    assert current['performance']['initiation_interval'] <= 85
    assert current['performance']['initiation_interval'] < baseline['performance']['initiation_interval']
    assert current['ooc']['resources']['LUT'] < baseline['ooc']['resources']['LUT']
    for name, capacity in {'LUT': 216960, 'FF': 433920, 'DSP': 1824, 'BRAM_TILES': 480}.items():
        assert current['ooc']['resources'][name] / capacity < 0.20
    assert current['ooc']['timing']['target_clock_ns'] == 5
    assert current['ooc']['timing']['wns_ns'] >= 0
    assert current['ooc']['timing']['tns_ns'] == 0
    assert current['ooc']['source_closure_sha256'] == current['source_closure_sha256']
    assert current['ooc']['manifest_sha256'] == current['manifest_sha256']
    for filename, expected in current['ooc']['report_files'].items():
        assert digest(ROOT / 'final_f12/ooc' / filename) == expected
    corpora = read(ROOT / 'final_f12/manifest.json')['verification']['corpora']
    assert corpora['built_in']['sample_count'] == 96
    assert corpora['supplied']['sample_count'] == 1000


def test_legacy_sources_and_qualification_artifacts_are_reproducible():
    comparisons = read(ROOT / 'legacy-comparison.json')
    assert len(comparisons) == 19
    assert all(row['identical_to_aria171'] and row['firmware_files'] == 86 for row in comparisons)
    for item in read(ROOT / 'calibration.json')['measurements']:
        assert digest(ROOT / item['hls_report_file']) == item['hls_report_sha256']
    reproduction = read(ROOT / 'reproducibility.json')
    for project in ('final_f12', 'final_c3', 'final_heldout'):
        assert reproduction[project]['firmware_sources_identical']
        assert reproduction[project]['selected_candidate_identical']
        assert reproduction[project]['compilation_inputs_identical']
        assert reproduction[project]['original_source_closure'] == reproduction[project]['repeat_source_closure']
    for filename, expected in read(ROOT / 'checksums.json').items():
        assert digest(ROOT / filename) == expected
