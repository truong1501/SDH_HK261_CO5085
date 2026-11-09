import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from src.dataset import SimpleImageDataset, get_dataloader
from src.diffusion import GaussianDiffusion
from src.dit import DiT
from src.unet import UNetDDPM


class PipelineTests(unittest.TestCase):
    def test_models_learn_and_sample(self):
        torch.set_num_threads(2)
        for model in [UNetDDPM(base_channels=8), DiT(input_size=8, hidden_size=24, depth=2, num_heads=4)]:
            with self.subTest(model=type(model).__name__):
                diffusion = GaussianDiffusion(timesteps=4, schedule="cosine", device="cpu")
                x = torch.randn(2, 3, 8, 8).clamp(-1, 1)
                noise = torch.randn_like(x)
                t = torch.tensor([0, 3])
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                before = [p.detach().clone() for p in model.parameters()]
                for _ in range(2):
                    optimizer.zero_grad()
                    predicted = model(diffusion.q_sample(x, t, noise), t)
                    self.assertEqual(predicted.shape, x.shape)
                    loss = (predicted-noise).square().mean()
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    optimizer.step()
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
                result, steps = diffusion.sample(model, x.shape, save_intermediate=True)
                self.assertTrue(torch.isfinite(result).all())
                self.assertLessEqual(result.abs().max().item(), 1)
                self.assertEqual([t for t, _ in steps], [3, 2, 1, 0])

    def test_split_and_small_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(20):
                Image.new("RGB", (10, 12), (i, 100, 200)).save(root/f"{i:06d}.jpg")
            sets = [SimpleImageDataset(root, image_size=8, split=s) for s in ["train", "val", "test"]]
            paths = [set(d.image_paths) for d in sets]
            self.assertEqual([len(p) for p in paths], [16, 2, 2])
            self.assertFalse(paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
            self.assertTrue(torch.equal(sets[1][0], sets[1][0]))
            self.assertEqual(sets[0].image_paths, SimpleImageDataset(root, image_size=8).image_paths)
            loader = get_dataloader(root, batch_size=128, num_workers=0)
            self.assertEqual(len(loader), 1)
            self.assertEqual(next(iter(loader)).shape[0], 16)
            partition = root/"list_eval_partition.txt"
            partition.write_text("".join(f"{i:06d}.jpg {i%3}\n" for i in range(20)))
            official = SimpleImageDataset(root, split="test", partition_file=partition)
            self.assertTrue(all(int(p.stem)%3 == 2 for p in official.image_paths))
            with self.assertRaises(ValueError):
                SimpleImageDataset(root/"missing")

    def test_clean_posterior_and_mixed_timesteps(self):
        diffusion = GaussianDiffusion(timesteps=4, schedule="cosine", device="cpu")
        clean = torch.full((2,3,8,8), .25)
        noise = torch.randn_like(clean)
        t = torch.tensor([0,3])
        noisy = diffusion.q_sample(clean, t, noise)
        class Oracle(torch.nn.Module):
            def forward(self, x, t):
                return noise
        result = diffusion.p_sample(Oracle(), noisy, t)
        self.assertTrue(torch.allclose(result[0], clean[0], atol=1e-5))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
