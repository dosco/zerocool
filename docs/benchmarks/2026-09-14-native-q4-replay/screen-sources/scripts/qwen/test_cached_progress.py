import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from cached_progress import summarize


class CachedProgressTest(unittest.TestCase):
    def events(self):
        events=[('model_load','begin',0,0,{}),('model_load','end',10,10,{}),
                ('reference_priming','begin',12,0,{'completed_tokens':0}),
                ('reference_priming','progress',20,8,{'completed_tokens':512,'active_tokens':512}),
                ('reference_priming','interrupted',30,18,{'phase_incomplete':True,'error':'cancelled'})]
        return [dict(kind='cached_replay_progress_v1',sequence=i+1,monotonic_ns=100+t,elapsed_ns=t,
                     phase_elapsed_ns=elapsed,phase=phase,event=event,identity={'build':'build'},details=details)
                for i,(phase,event,t,elapsed,details) in enumerate(events)]

    def query(self,rows,tail=b''):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'progress.jsonl'
            path.write_bytes(b''.join(json.dumps(row).encode()+b'\n' for row in rows)+tail)
            return summarize(path,'build')

    def test_interruption_preserves_last_work_and_elapsed_phase(self):
        result=self.query(self.events())
        self.assertFalse(result['complete']);self.assertEqual(result['status'],'interrupted')
        self.assertEqual(result['completed_phase_ns'],{'model_load':10})
        self.assertEqual(result['unfinished_phase'],'reference_priming')
        self.assertEqual(result['unfinished_phase_elapsed_ns'],18)
        self.assertEqual(result['latest_details']['completed_tokens'],512)
        self.assertEqual(result['last_elapsed_ns'],30)

    def test_partial_and_missing_traces_do_not_invent_completion(self):
        rows=self.events()[:-1]
        for tail in (b'{"sequence":',json.dumps(self.events()[-1]).encode()):
            result=self.query(rows,tail)
            self.assertTrue(result['partial_final_line_ignored']);self.assertEqual(result['status'],'unfinished')
            self.assertEqual(result['events'],4)
        self.assertFalse(self.query([])['available'])
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(summarize(Path(temp)/'missing')['available'])

    def test_repeated_phase_durations_and_complete_state(self):
        rows=self.events()[:2]
        rows+= [dict(rows[0],sequence=3,monotonic_ns=112,elapsed_ns=12),
                dict(rows[1],sequence=4,monotonic_ns=117,elapsed_ns=17,phase_elapsed_ns=5),
                dict(rows[1],sequence=5,monotonic_ns=118,elapsed_ns=18,phase_elapsed_ns=6,
                     event='complete',details={'phase_incomplete':False})]
        result=self.query(rows)
        self.assertEqual(result['completed_phase_ns']['model_load'],15)
        self.assertTrue(result['complete']);self.assertIsNone(result['unfinished_phase'])
        with self.assertRaises(ValueError):self.query(rows,b'{')

    def test_rejects_changed_identity_missing_events_and_false_completion(self):
        changes=[lambda r:r[2].update(identity={'build':'other'}),lambda r:r.pop(1),
                 lambda r:r[3].update(phase='other'),lambda r:r[3].update(elapsed_ns=-1),
                 lambda r:r[3].update(phase_elapsed_ns=9),lambda r:r[4].update(event='complete'),
                 lambda r:r.append(r[-1]),lambda r:r[4]['details'].update(phase_incomplete=False),
                 lambda r:r.__setitem__(0,[]),lambda r:r[0].update(sequence=True)]
        for change in changes:
            rows=copy.deepcopy(self.events());change(rows)
            with self.assertRaises(ValueError):self.query(rows)

    def test_native_load_failure_is_flushed_and_existing_trace_is_preserved(self):
        binary=Path(__file__).resolve().parents[2]/'build/qwen/bin/freellm'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);tokens=root/'tokens.json';tokens.write_text('[760,369]')
            trace=root/'progress.jsonl'
            command=[str(binary),'bench','--cached-token-replay','--model',str(root/'missing'),
                     '--memory-gb','12','--tokens-file',str(tokens),'--cached-progress',str(trace)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            report=summarize(trace)
            self.assertEqual(report['status'],'failed');self.assertEqual(report['last_phase'],'model_load')
            self.assertFalse(report['complete']);self.assertEqual(report['events'],2)
            original=trace.read_bytes()
            again=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(again.returncode,0);self.assertIn('must be new',again.stderr)
            self.assertEqual(trace.read_bytes(),original)

    def test_native_progress_is_cached_bench_only_and_requires_separate_output(self):
        binary=Path(__file__).resolve().parents[2]/'build/qwen/bin/freellm'
        for args,message in [(['run','--cached-progress','x'],'cached progress requires'),
                             (['bench','--cached-progress','x'],'cached progress requires'),
                             (['bench','--cached-token-replay','--tokens-file','x','--cached-progress','x','--json','x'],'separate output')]:
            result=subprocess.run([str(binary),*args],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0);self.assertIn(message,result.stderr)


if __name__=='__main__':unittest.main()
