import tempfile
import unittest

import torch
from torch import nn

from benchmarks.ht_xpu.qwen_sft import (
    IGNORE_INDEX,
    QwenSFTConfig,
    QwenSFTWorkload,
    deterministic_dolly_split,
    encode_response_only,
)


class TinyTokenizer:
    eos_token = "~"
    eos_token_id = 9
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [1] if add_special_tokens else []
        ids.extend(2 + (ord(char) % 7) for char in text)
        return {"input_ids": ids}


class TinyDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed):
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self), generator=generator).tolist()
        return TinyDataset([self.rows[index] for index in order])

    def select(self, indices):
        return TinyDataset([self.rows[index] for index in indices])


class TinyLM(nn.Module):
    def __init__(self, vocab_size=16, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.head(self.embedding(input_ids))
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        return type("Output", (), {"loss": loss, "logits": logits})


class TestQwenSFT(unittest.TestCase):
    def test_fixed_benchmark_defaults(self):
        self.assertEqual(QwenSFTConfig(training="lora").sequence_length, 512)
        self.assertEqual(QwenSFTConfig(training="lora").gradient_accumulation_steps, 4)
        self.assertEqual(QwenSFTConfig(training="full").sequence_length, 256)
        self.assertEqual(QwenSFTConfig(training="full").gradient_accumulation_steps, 8)

    def test_response_only_mask_and_fixed_shape(self):
        encoded = encode_response_only(
            {"instruction": "Do it", "context": "Here", "response": "OK"},
            TinyTokenizer(),
            48,
        )
        self.assertEqual(tuple(encoded["input_ids"].shape), (48,))
        supervised = encoded["labels"] != IGNORE_INDEX
        self.assertGreater(int(supervised.sum()), 0)
        self.assertTrue(torch.all(encoded["attention_mask"][~supervised] >= 0))
        padding = encoded["attention_mask"] == 0
        self.assertTrue(torch.all(encoded["labels"][padding] == IGNORE_INDEX))

    def test_deterministic_disjoint_split(self):
        data = TinyDataset(range(20))
        train_a, eval_a = deterministic_dolly_split(
            data, seed=7, train_examples=10, eval_examples=4
        )
        train_b, eval_b = deterministic_dolly_split(
            data, seed=7, train_examples=10, eval_examples=4
        )
        self.assertEqual(train_a.rows, train_b.rows)
        self.assertEqual(eval_a.rows, eval_b.rows)
        self.assertFalse(set(train_a.rows) & set(eval_a.rows))

    def test_full_training_and_checkpoint_round_trip(self):
        config = QwenSFTConfig(
            training="full",
            device="cpu",
            sequence_length=24,
            gradient_accumulation_steps=2,
        )
        model = TinyLM()
        workload = QwenSFTWorkload(config, model=model, tokenizer=TinyTokenizer())
        batch = workload.prepare_example({"instruction": "a", "response": "answer"})
        initial = model.head.weight.detach().clone()
        self.assertFalse(workload.train_step(batch)["optimizer_step"])
        self.assertTrue(workload.train_step(batch)["optimizer_step"])
        self.assertFalse(torch.equal(initial, model.head.weight))

        with tempfile.TemporaryDirectory() as directory:
            path = workload.save_checkpoint(directory)
            expected = model.head.weight.detach().clone()
            with torch.no_grad():
                model.head.weight.zero_()
            workload.load_checkpoint(directory)
            self.assertTrue(torch.equal(expected, model.head.weight))
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
