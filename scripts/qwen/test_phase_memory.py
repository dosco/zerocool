import copy
import unittest
from benchmark_exact import validate_phase_memory, config_args, is_original_control


class PhaseMemoryTest(unittest.TestCase):
    def fixture(self):
        prompt=dict(limit_bytes=1000,planned_bytes=990,expert_slots=32)
        generation=dict(limit_bytes=1000,planned_bytes=999,expert_slots=64)
        phase=dict(policy='reclaim',phase='generation',prompt_plan=prompt,generation_plan=generation,
                   pressure_resizes=0,transition_count=0,transitions=[])
        before=dict(phase_memory=phase,memory_plan=generation)
        after=copy.deepcopy(before)
        after['phase_memory'].update(transition_count=2,transitions=[
            dict(sequence=1,reason='ingest_panel',before=generation,after=prompt,live_buffer_bytes=900,scratch_after=[]),
            dict(sequence=2,reason='ingest_complete',before=prompt,after=generation,live_buffer_bytes=900,
                 scratch_after=[dict(allocated_bytes=0),dict(allocated_bytes=0)])])
        return dict(before=before,after=after)

    def test_declared_transitions_and_noop(self):
        row=self.fixture();config=dict(phase_memory='reclaim',prefill_pipeline='double')
        validate_phase_memory(row,config,1000)
        row['after']=copy.deepcopy(row['before']);validate_phase_memory(row,config,1000)
        self.assertFalse(is_original_control(config))
        self.assertIn('--phase-memory',config_args(config))

    def test_incomplete_or_unsafe_transitions_fail(self):
        mutations=[lambda p:p.update(pressure_resizes=1),lambda p:p.update(transition_count=3),
            lambda p:p['transitions'][0].update(reason='pressure'),
            lambda p:p['transitions'][1].update(live_buffer_bytes=1001),
            lambda p:p['transitions'][1]['scratch_after'][0].update(allocated_bytes=1),
            lambda p:p['generation_plan'].update(planned_bytes=1001),
            lambda p:p.update(phase='prompt')]
        for mutate in mutations:
            row=self.fixture();mutate(row['after']['phase_memory'])
            with self.assertRaises(ValueError):validate_phase_memory(row,dict(phase_memory='reclaim',prefill_pipeline='double'),1000)

    def test_capture_schedules_cover_distinct_phases_and_layers(self):
        from capture_phase_inputs import CASES
        self.assertEqual({p for p,_,_ in CASES},{'prefill','append','decode'})
        self.assertTrue({0,23,30,31,47}.issubset({l for _,l,_ in CASES}))
        self.assertEqual(len(CASES),len(set(CASES)))

    def test_fixed_cannot_hide_pressure(self):
        row=dict(before=dict(phase_memory=dict(pressure_resizes=0)),after=dict(phase_memory=dict(pressure_resizes=1)))
        with self.assertRaises(ValueError):validate_phase_memory(row,{},1000)


class HeldoutPolicyTest(unittest.TestCase):
    def report(self):
        matrix=dict(K=64,N=8,rows=128,group=64,bits=8,fused=False,gathered=False)
        rows=[]
        for rep in range(5):
            for tile in [1,2,4,8]:
                rows.append(dict(matrix=matrix,tile=tile,repetition=rep,exact=True,case='fixture',wall_ns=1000 if tile==1 else 800))
            rows.append(dict(matrix=matrix,tile=8,output_rows=4,gate_pair=False,repetition=rep,exact=True,case='fixture',wall_ns=400))
        return dict(kind='captured_operator_screen',exact=True,build_fingerprint='build',artifact_revision='artifact',measurements=rows)

    def test_select_and_validate_without_reselection(self):
        from select_shape_rules import select,validate_heldout
        report=self.report();policy=select(report)
        self.assertEqual(policy['rules'][0]['output_rows'],4)
        self.assertTrue(validate_heldout(policy,report)['passed'])
        for row in report['measurements']:
            if row.get('output_rows')==4:row['wall_ns']=1200
        self.assertFalse(validate_heldout(policy,report)['passed'])

    def test_missing_coverage_and_identity_fail(self):
        from select_shape_rules import select,validate_heldout,merge_reports
        report=self.report();policy=select(report)
        empty=dict(report,measurements=[])
        self.assertFalse(validate_heldout(policy,empty)['passed'])
        with self.assertRaises(ValueError):validate_heldout(policy,dict(report,build_fingerprint='other'))
        with self.assertRaises(ValueError):merge_reports([report,dict(report,artifact_revision='other')])

    def test_heldout_rejects_unpaired_samples(self):
        from select_shape_rules import select,validate_heldout
        for variant in [(1,1),(8,1),(8,4)]:
            report=self.report();policy=select(report)
            sample=next(row for row in report['measurements']
                        if (row['tile'],row.get('output_rows',1))==variant)
            report['measurements'].append(dict(sample,repetition=5,wall_ns=100000))
            with self.subTest(variant=variant),self.assertRaises(ValueError):
                validate_heldout(policy,report)

    def test_heldout_rejects_gaps_in_repetitions(self):
        from select_shape_rules import select,validate_heldout
        for missing in (0,2):
            report=self.report();policy=select(report)
            for row in report['measurements']:
                if row['repetition']>=missing:row['repetition']+=1
            with self.subTest(missing=missing),self.assertRaises(ValueError):
                validate_heldout(policy,report)

    def test_failed_heldout_cli_returns_failure(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from select_shape_rules import select
        report=self.report();policy=select(report)
        for row in report['measurements']:
            if row.get('output_rows')==4:row['wall_ns']=1200
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);r=root/'report.json';p=root/'policy.json';out=root/'result.json'
            r.write_text(json.dumps(report));p.write_text(json.dumps(policy))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('select_shape_rules.py')),str(r),
                '--validate-policy',str(p),'--output',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertFalse(json.loads(out.read_text())['passed'])

if __name__=='__main__':unittest.main()
