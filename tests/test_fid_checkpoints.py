"""FID snapshots between recovery intervals remain exactly resumable."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import main


class FIDCheckpointTests(unittest.TestCase):
    def test_fid_checkpoint_and_resume_between_recovery_intervals(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'dummy_weights'
            weights.write_bytes(b'test')
            fake_fid = SimpleNamespace(WEIGHTS=weights, CIFAR10FID=lambda *a, **k:
                SimpleNamespace(evaluate_generator=lambda *a, **k: 12.5))
            data = torch.arange(6 * 3072).reshape(6, 3072).remainder(256).byte()
            labels = torch.zeros(6, dtype=torch.long)

            def run(output, resume=False):
                argv = ['main.py', '--dataset', 'cifar10', '--architecture', 'mlp',
                        '--latent-dim', '8', '--hidden-dims', '8', '--target-size', '0',
                        '--target-batch-size', '2', '--particles', '3', '--eval-size', '4',
                        '--steps', '5', '--save-every', '2', '--checkpoint-every', '3',
                        '--keep-checkpoint-every', '5', '--ema-decay', '.9', '--save-ema-at-fid',
                        '--threads', '1', '--device', 'cpu', '--output', str(output)]
                if resume:
                    argv += ['--resume', str(output / 'checkpoint_latest.pt')]
                with patch('sys.argv', argv), patch.dict('sys.modules', fid=fake_fid), \
                        patch.object(main, 'load_dataset', return_value=(
                            Path('/cifar-10-python.tar.gz'), (data, labels), (data, labels))), \
                        patch.object(main, 'save_grid'), patch.object(main, 'save_trajectory'), \
                        patch.object(main, 'plot_mmd'):
                    main.main()
                return torch.load(output / 'checkpoint_latest.pt', weights_only=False)

            full = run(root / 'full')
            original_step = main.generator_step
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('stop after FID at step two')
                return original_step(*args, **kwargs)

            with patch.object(main, 'generator_step', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'stop after FID'):
                    run(root / 'resumed')
            saved = torch.load(root / 'resumed/checkpoint_latest.pt', weights_only=False)
            self.assertEqual(saved['update_count'], 2)
            self.assertEqual(saved['fid_history'][-1], (2, 12.5))
            self.assertTrue(saved['optimizer']['state'])
            self.assertTrue(saved['config']['checkpoint_at_fid'])
            self.assertFalse((root / 'resumed/checkpoint_step_000002.pt').exists())
            ema = torch.load(root / 'resumed/fid_checkpoints/ema_step_000002.pt', weights_only=False)
            self.assertEqual(ema['checkpoint_kind'], 'evaluation_only_ema')
            self.assertEqual(ema['update_count'], 2)
            self.assertEqual(ema['fid'], 12.5)
            self.assertNotIn('optimizer', ema)
            for name in saved['ema_model']:
                torch.testing.assert_close(ema['ema_model'][name], saved['ema_model'][name], rtol=0, atol=0)
            resumed = run(root / 'resumed', resume=True)
            for key in ('model', 'ema_model'):
                for name in full[key]:
                    torch.testing.assert_close(full[key][name], resumed[key][name], rtol=0, atol=0)
            self.assertEqual(full['fid_history'], resumed['fid_history'])
            self.assertEqual(json.loads((root / 'resumed/fid_metrics.json').read_text())[-1]['iteration'], 5)
            self.assertTrue((root / 'resumed/checkpoint_step_000005.pt').exists())
            self.assertEqual(sorted(p.name for p in (root / 'resumed/fid_checkpoints').glob('*.pt')),
                [f'ema_step_{step:06d}.pt' for step in (0, 2, 4, 5)])


if __name__ == '__main__':
    unittest.main()
