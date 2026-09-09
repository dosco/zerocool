import copy
import unittest
from benchmark_exact import configurations, config_args, workloads, paired_interval, summarize, validate, fixed_sampling
from tune_exact import choose
from qualify_exact_sessions import compare, check_prime
from check_soak import evaluate
import test_panel_benchmark as panel_test


class ExactBenchmarkTest(unittest.TestCase):
    def fixture(self):
        report=panel_test.PanelComparisonTest().fixture();row=report["runs"][0]
        row.update(runtime_cache_state="empty_at_process_start",repetition=0,profiling_enabled=False,
                   finish_reason="length",pending_tokens_ingested=0,decode_ms=25500,decode_wall_ms=25600,request_ms=26600,wall_tokens_per_second=9.96,token_latency_ms=[100]*255,p95_token_ms=100,phases={})
        report["sampling"]=dict(temperature=0,top_k=20,top_p=0.95,seed=0)
        row["after"].update(completion_pipeline=True,chunk_tokens=128,ready_group=4,io_workers=8)
        row["after"]["metal"]["kernels"]={"policy":"reference","token_tile":1,"gdn":"original","gdn_rows":4,"gdn_block":8}
        row["after"]["memory_plan"]["panel_tokens"]=512
        return report

    def test_prime_counts_are_separate(self):
        cases=workloads([1,2,3],256)
        prime,append=cases["append_128"]
        self.assertTrue(prime["prime"]);self.assertEqual(len(prime["tokens"]),4096)
        self.assertNotIn("max_tokens",prime)
        self.assertEqual(len(append["tokens"]),128)
        self.assertEqual(len(cases["prompt_2k"][0]["tokens"]),2048)

    def test_native_sampling_float_serialization(self):
        settings=dict(temperature=0,top_k=20,top_p=0.949999988079071,seed=0)
        self.assertTrue(fixed_sampling(settings))
        self.assertFalse(fixed_sampling(dict(settings,top_p=0.95000005)))
        self.assertFalse(fixed_sampling(dict(settings,seed=1)))
        self.assertFalse(fixed_sampling(None))

    def test_reject_warm_diagnostic_reduced_and_changed_runs(self):
        report=self.fixture();config=configurations()[0]
        expected={"build":"native","revision":"revision"}
        validate(report,config,expected,8*1024**3,256)
        changes=[lambda r:r["runs"][0].update(runtime_cache_state="retained"),
                 lambda r:r["runs"][0].update(profiling_enabled=True),
                 lambda r:r["runs"][0].update(repetition=1),
                 lambda r:r["sampling"].update(temperature=1),
                 lambda r:r["runs"][0]["after"].update(io_workers=4),
                 lambda r:r["runs"][0]["after"]["metal"]["kernels"].update(token_tile=8),
                 lambda r:r["runs"][0]["after"]["metal"]["kernels"].update(gdn_rows=8),
                 lambda r:r["runs"][0]["after"]["memory_plan"].update(limit_bytes=7*1024**3)]
        for change in changes:
            bad=copy.deepcopy(report);change(bad)
            with self.assertRaises(ValueError):validate(bad,config,expected,8*1024**3,256)

    def test_paired_confidence_and_configuration_validation(self):
        bounds=paired_interval([0.8]*5)
        self.assertEqual(bounds["low"],0.8);self.assertEqual(bounds["high"],0.8)
        with self.assertRaises(ValueError):paired_interval([0.8]*4)
        with self.assertRaises(ValueError):config_args({"name":"test","unknown":1})

    def test_partial_workload_coverage_cannot_qualify(self):
        rows=[dict(configuration=name,name="prompt_2k",pair=p,ttft_ms=latency,decode_ms_per_token=latency)
              for p in range(5) for name,latency in [("reference",100),("precompute",80)]]
        result=summarize(rows,configurations()[:2])[0]
        self.assertFalse(result["required_workloads_complete"])
        self.assertFalse(result["latency_gate_passed"])

    def test_tuning_balances_latency_and_keeps_ties(self):
        rows=[]
        for name,ttft,decode in [("baseline",100,10),("prefill_only",80,20),("balanced",90,9)]:
            rows.append(dict(configuration=name,name="prompt_2k",ttft_ms=ttft,decode_ms_per_token=decode))
        self.assertEqual(choose(dict(measurements=rows))[0],"balanced")

    def test_session_qualification_rejects_incomplete_or_changed_state(self):
        stage=dict(layers=[{}]*48,routes=[{}]*48,logits_sha256="identical")
        stats=dict(metal=dict(build_fingerprint="native"),diagnostic_stream_trunk=False,memory_plan=dict(panel_tokens=512))
        reference=dict(passed=True,runs=[dict(panel=512,stages=[stage],continued_statistics=stats)])
        self.assertTrue(compare(reference,copy.deepcopy(reference),"native"))
        for change in [lambda r:r.update(passed=False),
                       lambda r:r["runs"][0]["stages"][0].update(logits_sha256="changed"),
                       lambda r:r["runs"][0]["continued_statistics"].update(diagnostic_stream_trunk=True),
                       lambda r:r["runs"][0]["continued_statistics"]["memory_plan"].update(panel_tokens=256)]:
            candidate=copy.deepcopy(reference);change(candidate)
            with self.assertRaises(ValueError):compare(reference,candidate,"native")

    def test_prime_qualification_checks_actual_reuse_and_sampling(self):
        stats=dict(metal=dict(build_fingerprint="native"),diagnostic_stream_trunk=False)
        prime=dict(finish_reason="primed",output_tokens=0,prompt_tokens=5,after=stats)
        continued=dict(reused_tokens=5,prefill_tokens=3,pending_tokens_ingested=0,output_tokens=2,
                       output_token_ids=[1,2],after=stats)
        fresh=dict(continued,reused_tokens=0)
        report=dict(complete=True,runs=[prime,continued,fresh])
        self.assertEqual(check_prime(report,"native",5,3),[1,2])
        for change in [lambda r:r["runs"][0].update(output_tokens=1),
                       lambda r:r["runs"][1].update(pending_tokens_ingested=1),
                       lambda r:r["runs"][2].update(output_token_ids=[1,3])]:
            bad=copy.deepcopy(report);change(bad)
            with self.assertRaises(ValueError):check_prime(bad,"native",5,3)

    def test_soak_rejects_incomplete_evidence(self):
        with self.assertRaises(ValueError):evaluate(dict(complete=True,benchmark_elapsed_ns=1199*10**9,runs=[]))
        machine=dict(build_fingerprint="build",device="Apple M1 Pro",physical_bytes=32*1024**3)
        rows=[]
        for _ in range(6):
            rows.append(dict(request_ms=240000,profiling_enabled=False,
                after=dict(metal=machine,artifact_revision="artifact",diagnostic_stream_trunk=False,
                    memory_plan=dict(limit_bytes=8*1024**3,expert_slots=300),
                    process=dict(physical_footprint_bytes=7*1024**3,system_swap_used_bytes=1024**3))))
        report=dict(complete=True,benchmark_elapsed_ns=1440*10**9,runs=rows)
        self.assertTrue(evaluate(report)["passed"])
        bad=copy.deepcopy(report);bad["runs"][-1]["after"]["process"]["physical_footprint_bytes"]+=512*1024**2
        self.assertFalse(evaluate(bad)["passed"])

if __name__=="__main__":unittest.main()
