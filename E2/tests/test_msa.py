"""Đối chiếu MSA bằng gradient đầu ra không đồng đều; không ghi đè kết quả đã lưu."""
import unittest
import torch
from src.attention import ScratchMSA, TorchMSA
from src.check_msa import copy_weights, check
from src.models import build_model


class MsaTests(unittest.TestCase):
    def test_reference_matches_input_and_parameter_gradients(self):
        torch.manual_seed(42)
        reference = TorchMSA(16, 4).eval()
        scratch = ScratchMSA(16, 4).eval()
        copy_weights(reference, scratch)
        x1 = torch.randn(2, 5, 16, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        y1, a1 = reference(x1, return_attn=True)
        y2, a2 = scratch(x2, return_attn=True)
        upstream = torch.randn_like(y1)
        (y1 * upstream).sum().backward()
        (y2 * upstream).sum().backward()
        for actual, expected in [(y2, y1), (a2, a1), (x2.grad, x1.grad),
                                 (scratch.qkv.weight.grad, reference.mha.in_proj_weight.grad),
                                 (scratch.qkv.bias.grad, reference.mha.in_proj_bias.grad),
                                 (scratch.out_proj.weight.grad, reference.mha.out_proj.weight.grad),
                                 (scratch.out_proj.bias.grad, reference.mha.out_proj.bias.grad)]:
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        self.assertTrue(check()["passed"])

    def test_head_dimension_must_divide_embedding(self):
        with self.assertRaises(ValueError):
            ScratchMSA(15, 4)

    def test_all_tokenizers_preserve_classifier_contract(self):
        expected = {"patch4": 50, "patch7s3": 65, "row": 29, "cnnstem": 50}
        images = torch.randn(2, 1, 28, 28)
        for tokenizer, length in expected.items():
            with self.subTest(tokenizer=tokenizer), torch.no_grad():
                model = build_model(tokenizer, "scratch").eval()
                logits, attention = model(images, return_attn=True)
                self.assertEqual(tuple(logits.shape), (2, 10))
                self.assertEqual(len(attention), 4)
                for weights in attention:
                    self.assertEqual(tuple(weights.shape), (2, 4, length, length))
                    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, length))


if __name__ == "__main__":
    unittest.main()
