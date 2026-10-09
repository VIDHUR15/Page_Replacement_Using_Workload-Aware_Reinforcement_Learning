"""Checks for the new evaluation wrapper and preservation of original behavior."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import benchmark
import evaluate


class BenchmarkTests(unittest.TestCase):
    def test_original_source_bytes(self):
        root=Path(__file__).resolve().parents[1]
        hashes=json.loads((root/'ORIGINAL_SOURCE_SHA256.json').read_text())
        self.assertEqual(len(hashes),9)
        for name,expected in hashes.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),expected,name)

    def test_trial_zero_matches_original_all_workloads(self):
        for name in benchmark.WORKLOADS:
            original=evaluate.evaluate_workload(name)
            updated,_=benchmark.run_trial(name,20,0,benchmark.Config())
            for old,new in zip(original,updated):
                self.assertEqual(old['policy'],new['source_policy'])
                for key in ('accesses','hits','faults','hit_rate','fault_rate','gap_to_opt_pct','policy_usage'):
                    self.assertEqual(old.get(key),new.get(key),(name,old['policy'],key))

    def test_pairing_and_random_trace_reproducibility(self):
        config=benchmark.Config(length=80,episodes=2)
        rows,manifest=benchmark.run_trial('mixed',5,3,config)
        again,other=benchmark.run_trial('mixed',5,3,config)
        self.assertEqual(manifest,other)
        self.assertEqual(len({r['trace_sha256'] for r in rows}),1)
        self.assertNotIn(manifest['seeds']['evaluation'],manifest['seeds']['training'])
        for a,b in zip(rows,again):
            for key in ('faults','hits','policy_usage','trace_sha256'):
                self.assertEqual(a.get(key),b.get(key))
        for r in rows:
            self.assertEqual(r['hits']+r['faults'],config.length)

    def test_deterministic_overlap_is_reported(self):
        config=benchmark.Config(length=100,episodes=3)
        for name in ('sequential','looping'):
            _,m=benchmark.run_trial(name,5,0,config)
            _,n=benchmark.run_trial(name,5,1,config)
            self.assertTrue(m['deterministic_workload'])
            self.assertEqual(m['training_trace_overlap_count'],3)
            self.assertEqual(m['evaluation_trace_sha256'],n['evaluation_trace_sha256'])
            self.assertNotEqual(m['seeds']['agent'],n['seeds']['agent'])

    def test_summaries_and_pairwise_counts(self):
        config=benchmark.Config(length=50,episodes=1)
        rows,_=benchmark.run_trial('random',3,0,config)
        single=benchmark.summarize(rows)
        self.assertTrue(all(r['hit_rate_std'] is None for r in single))
        additional,_=benchmark.run_trial('random',3,1,config)
        rows+=additional
        summary=benchmark.summarize(rows)
        self.assertEqual(len(summary),8)
        self.assertTrue(all(r['trials']==2 and r['hit_rate_std'] is not None for r in summary))
        for r in benchmark.paired_comparisons(rows):
            self.assertEqual(r['wins']+r['ties']+r['losses'],2)

    def test_invalid_parameters(self):
        for config in (benchmark.Config(length=0),benchmark.Config(episodes=0),
                       benchmark.Config(min_commit=0),benchmark.Config(page_range=1)):
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            benchmark.run_trial('random',0,0,benchmark.Config())
        with self.assertRaises(ValueError):
            benchmark.run_trial('random',2,-1,benchmark.Config())

    def test_cli_exports_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run'
            args=['--workloads','mixed','--seeds','0','1','--capacities','3',
                  '--length','50','--episodes','1','--output',str(p)]
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.main(args)
            self.assertEqual(json.loads((p/'config.json').read_text())['status'],'complete')
            with (p/'raw_results.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),16)
            self.assertTrue((p/'summary.csv').is_file())
            self.assertTrue((p/'paired_comparisons.csv').is_file())
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                benchmark.main(args)


if __name__=='__main__':
    unittest.main()
