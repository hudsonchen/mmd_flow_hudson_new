import importlib.util
from pathlib import Path
import unittest
from checkpoint_policy import checkpoint_due


class CheckpointPolicyTests(unittest.TestCase):
    def test_recovery_cadence_and_final(self):
        total=750684
        self.assertTrue(checkpoint_due(3,total,10000))
        for step in [1000,4000,5000,9000,15000]:
            self.assertFalse(checkpoint_due(step,total,10000))
        for step in [10000,20000,total]:
            self.assertTrue(checkpoint_due(step,total,10000))

    def test_best_between_periodic_saves_requires_current_checkpoint(self):
        saves=[]
        latest=None
        best=None
        for step in range(1,20001):
            if checkpoint_due(step,20000,10000):
                saves.append(step);latest=step
            if step in [5000,15000]:
                if not checkpoint_due(step,20000,10000):
                    saves.append(step);latest=step
                best=latest
                self.assertEqual(best,step)
        self.assertEqual(saves,[3,5000,10000,15000,20000])

    def test_resume_change_is_explicit_and_training_settings_protected(self):
        path=Path(__file__).resolve().parents[1]/'scripts/cs_cluster/run_imagenet64_jit.py'
        spec=importlib.util.spec_from_file_location('checkpoint_runner',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old={'lr':.0002,'objective':'coupled','solver':'heun','required_gpu':'H100'}
        new={**old,'checkpoint_every':10000}
        with self.assertRaises(ValueError):module.validate_resume_config(old,new)
        self.assertFalse(module.validate_resume_config(old,new,allow_checkpoint_change=True))
        with self.assertRaises(ValueError):
            module.validate_resume_config(old,{**new,'lr':.1},allow_checkpoint_change=True)
        with self.assertRaises(ValueError):
            module.validate_resume_config(old,{**new,'objective':'flow_matching'},allow_checkpoint_change=True)


if __name__=='__main__':unittest.main()
