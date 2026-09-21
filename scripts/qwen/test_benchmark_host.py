import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_host as host
from qualification_evidence import ResourceBlocked


def fixture():
    return dict(kind='benchmark_host_preflight_v1',complete=True,build_fingerprint='build',
        model_loaded=False,gpu_used=False,
        host=dict(thermal_state=0,low_power_mode=False,power_source='AC Power',monotonic_ns=1))


class HostPreflightTest(unittest.TestCase):
    def test_good_host_and_explicit_low_power_or_thermal_blockers(self):
        self.assertTrue(host.observe(fixture(),'build')['clean_host'])
        for field,value,text in [('low_power_mode',True,'Low Power Mode'),('thermal_state',1,'thermal')]:
            raw=fixture();raw['host'][field]=value;result=host.observe(raw,'build')
            self.assertFalse(result['clean_host']);self.assertIn(text,result['reasons'][0])
        # Match the existing protocol: a stable battery source is allowed with LPM off.
        raw=fixture();raw['host']['power_source']='Battery Power'
        self.assertTrue(host.observe(raw,'build')['clean_host'])

    def test_missing_invalid_or_changed_identity_is_not_clean(self):
        for field,value in [('thermal_state',None),('thermal_state',False),('thermal_state',4),
            ('low_power_mode',None),('low_power_mode',0),('power_source',''),('monotonic_ns',0)]:
            raw=fixture();raw['host'][field]=value
            with self.assertRaises(ValueError):host.observe(raw,'build')
        for field in ('complete','model_loaded','gpu_used'):
            raw=fixture();raw[field]=not raw[field]
            with self.assertRaises(ValueError):host.observe(raw,'build')
        with self.assertRaises(ValueError):host.observe(fixture(),'changed-build')

    def test_blocked_preflight_is_persisted_without_running_a_model(self):
        raw=fixture();raw['host']['low_power_mode']=True
        with tempfile.TemporaryDirectory() as directory:
            class Experiment:
                out=Path(directory);frozen=dict(build='build');report={};commands=[];persisted=None
                def command(self,command,stem,limit):
                    self.commands.append((command,stem,limit));command[-1].write_text(json.dumps(raw))
                def persist(self):self.persisted=copy.deepcopy(self.report)
            exp=Experiment();probe=dict(binary=Path('/fake/host-check'))
            with self.assertRaisesRegex(ResourceBlocked,'Before validate-1: Low Power Mode'):
                host.preflight(exp,probe,'validate-1')
            self.assertEqual(len(exp.commands),1);self.assertEqual(exp.commands[0][2],5)
            self.assertEqual(exp.report,exp.persisted)
            self.assertEqual(exp.report['host_preflight'][0]['source'],'validate-1-host.json')

    def test_builder_uses_native_flags_without_production_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();native=root/'build/qwen';(native/'CMakeFiles/zerocool.dir').mkdir(parents=True)
            for folder in ('scripts/qwen','include/engine'):(root/folder).mkdir(parents=True)
            for name in ('benchmark_host.cpp','build_identity.py','qualification_evidence.py'):(root/'scripts/qwen'/name).write_text('fixture')
            source=root/'src/engine/model.cpp'
            (native/'compile_commands.json').write_text(json.dumps([dict(file=str(source),directory=str(native),
                command=f'/usr/bin/c++ -std=c++23 -fno-fast-math -o model.o -c {source}')]))
            (native/'CMakeFiles/zerocool.dir/link.txt').write_text('/usr/bin/c++ CMakeFiles/zerocool.dir/src/engine/main.cpp.o -o zerocool libzerocool_lib.a')
            (native/'libzerocool_lib.a').write_text('fixture')
            def compile(command,**kwargs):Path(command[command.index('-o')+1]).write_text('binary')
            with patch.object(host,'ROOT',root),patch.object(host,'build_fingerprint',return_value='build'),\
                    patch.object(host.subprocess,'run',side_effect=compile):
                proof=host.build_probe(root/'probe')
            self.assertTrue(proof['producer']['complete'])
            self.assertIn('-fno-fast-math',proof['producer']['compiler'])
            self.assertNotIn('CMakeFiles/zerocool.dir/src/engine/main.cpp.o',proof['producer']['linker'])
            self.assertLess(proof['producer']['linker'].index(str(root/'probe/host.o')),
                proof['producer']['linker'].index('libzerocool_lib.a'))


if __name__=='__main__':unittest.main()
