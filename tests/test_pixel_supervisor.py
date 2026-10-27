"""Decision and checkpoint preservation checks for autonomous run supervision."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.lsf.supervise_pixel_flow import assess, preserve_and_stop


class SupervisorTests(unittest.TestCase):
    def test_stops_only_three_material_regressions_after_minimum_budget(self):
        rows=[dict(update=5000*(i+1),fid=f) for i,f in enumerate([10.,12.,13.,12.])]
        self.assertEqual(assess(rows,{},[])['action'],'stop_regression')
        self.assertEqual(assess(rows[:3],{},[])['action'],'continue')
        rows[-1]['fid']=10.5
        self.assertEqual(assess(rows,{},[])['action'],'continue')
        rows[-1]['fid']=9.
        self.assertEqual(assess(rows,{},[])['action'],'continue')

    def test_plateau_and_improvement_continue(self):
        for scores in [[20.,15.,14.,12.],[10.,10.1,10.2,10.3]]:
            rows=[dict(update=5000*(i+1),fid=f) for i,f in enumerate(scores)]
            self.assertEqual(assess(rows,{},[],11.5)['action'],'continue')

    def test_nonfinite_is_not_silently_accepted(self):
        self.assertEqual(assess([],dict(loss=float('nan')),[])['action'],'stop_nonfinite')
        self.assertEqual(assess([dict(update=5000,fid=float('inf'))],{},[])['action'],'stop_nonfinite')

    def test_no_cancellation_without_preserved_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'training').mkdir()
            with patch('scripts.lsf.supervise_pixel_flow.subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError,'Cannot stop'):
                    preserve_and_stop(root,'123',root/'archive',dict(reason='test'))
                run.assert_not_called()

    def test_archives_before_cancelling_and_marks_only_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'training').mkdir()
            for name in ['best.pt','latest.pt']:(root/'training'/name).write_bytes(name.encode())
            def cancelled(*args,**kwargs):
                self.assertEqual((root/'archive/best.pt').read_bytes(),b'best.pt')
                self.assertEqual((root/'archive/latest.pt').read_bytes(),b'latest.pt')
                self.assertTrue((root/'archive/decision_before_stop.json').is_file())
                from subprocess import CompletedProcess
                return CompletedProcess(args[0],0,'terminated','')
            with patch('scripts.lsf.supervise_pixel_flow.subprocess.run',side_effect=cancelled):
                preserve_and_stop(root,'123',root/'archive',dict(reason='test'))
            self.assertTrue((root/'early_stop_decision.json').is_file())


if __name__ == '__main__':unittest.main()
