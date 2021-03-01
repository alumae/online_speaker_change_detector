import unittest
from online_scd.data import SCDDataset
import torch

class TestSCDDataset(unittest.TestCase):

    def test_dataset(self):
        dataset = SCDDataset("test/sample_dataset", extract_chunks=True, min_chunk_length=5.0, max_chunk_length=25.0, no_augment=True)
        self.assertEqual(len(dataset), 61)
        print(dataset[0])

        
    def test_collate(self):
        dataset = SCDDataset("test/sample_dataset", extract_chunks=True, min_chunk_length=5.0, max_chunk_length=25.0, no_augment=True)
        batch = dataset.collater([dataset[0], dataset[1]])
        print(batch["label"])

if __name__ == '__main__':
    unittest.main()