"""Standard-library smoke tests for the small Transformer example."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_transformer import CausalSelfAttention, CausalTransformer, DEFAULT_TEXT, Vocabulary, generate, make_batch


class TransformerSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.vocabulary = Vocabulary.from_text(DEFAULT_TEXT)
        self.model = CausalTransformer(
            vocab_size=len(self.vocabulary.itos),
            block_size=8,
            d_model=16,
            nhead=4,
            num_layers=1,
            dropout=0.0,
        )

    def test_forward_shape(self) -> None:
        tokens = self.vocabulary.encode("transfor").unsqueeze(0)
        logits = self.model(tokens)
        self.assertEqual(logits.shape, (1, 8, len(self.vocabulary.itos)))

    def test_attention_is_causal(self) -> None:
        attention = CausalSelfAttention(16, 4, 0.0).eval()
        original = torch.randn(1, 8, 16)
        changed = original.clone()
        changed[:, 5:] += 100
        with torch.no_grad():
            left = attention(original)
            right = attention(changed)
        self.assertTrue(torch.allclose(left[:, :5], right[:, :5], rtol=0, atol=1e-6))

    def test_invalid_dimensions_fail_before_forward(self) -> None:
        with self.assertRaises(ValueError):
            CausalTransformer(len(self.vocabulary.itos), 8, 15, 4, 1, 0.0)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 9, dtype=torch.long))

    def test_make_batch_shifts_targets_by_one_token(self) -> None:
        data = torch.arange(32, dtype=torch.long)
        inputs, targets = make_batch(data, batch_size=4, block_size=6, device=torch.device("cpu"))
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))

    def test_generation_appends_requested_token_count(self) -> None:
        prompt = self.vocabulary.encode("transform").unsqueeze(0)
        generated = generate(self.model, prompt, new_tokens=5, temperature=1.0)
        self.assertEqual(generated.shape, (1, prompt.size(1) + 5))


if __name__ == "__main__":
    unittest.main()
