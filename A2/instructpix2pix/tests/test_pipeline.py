import ast
from pathlib import Path
import unittest
from prepare_data import split_groups


class DataTests(unittest.TestCase):
    def test_grouped_split_has_no_source_leakage(self):
        keys = [str(i//2) for i in range(100)]
        split = split_groups(keys)
        groups = {s: {keys[i] for i in ids} for s, ids in split.items()}
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['test'] & (groups['train'] | groups['validation']))
        self.assertEqual(sorted(i for ids in split.values() for i in ids), list(range(100)))
        self.assertEqual(split, split_groups(keys))
        self.assertNotEqual(split, split_groups(keys, 43))

    def test_small_dataset_rejected(self):
        with self.assertRaises(ValueError):
            split_groups(['same']*100)

    def test_upstream_keeps_eight_channel_checkpoint(self):
        import torch
        from types import SimpleNamespace
        source = (Path(__file__).resolve().parents[1]/'train_instruct_pix2pix.py').read_text()
        module = ast.parse(source)
        main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        guard = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'unet.config.in_channels == 4')
        code = compile(ast.Module(body=[guard], type_ignores=[]), '<guard>', 'exec')
        class Fake:
            def __init__(self, channels):
                self.conv_in = torch.nn.Conv2d(channels, 4, 3, padding=1)
                self.config = SimpleNamespace(in_channels=channels)
            def register_to_config(self, **kwargs):
                self.config.in_channels = kwargs['in_channels']
        eight = Fake(8)
        original = eight.conv_in
        exec(code, dict(unet=eight, torch=torch, nn=torch.nn))
        self.assertIs(original, eight.conv_in)
        four = Fake(4)
        weight = four.conv_in.weight.detach().clone()
        bias = four.conv_in.bias.detach().clone()
        exec(code, dict(unet=four, torch=torch, nn=torch.nn))
        self.assertTrue(torch.equal(four.conv_in.weight[:, :4], weight))
        self.assertEqual(four.conv_in.weight[:, 4:].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(four.conv_in.bias, bias))
        self.assertEqual(four.config.in_channels, 8)


if __name__ == '__main__':
    unittest.main()
