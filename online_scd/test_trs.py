import unittest
from trs import Transcritpion
import torch

class TestTranscription(unittest.TestCase):

    def test_speech_section(self):
        transcription = Transcritpion("test/sample_dataset/71_ID117_344945.wav", "test/sample_dataset/71_ID117_344945.trs")
        speech_sections = transcription.get_speech_sections()
        self.assertEqual(len(speech_sections), 1)
        
        #print(speech_sections[0].relative_speaker_change_points)
        self.assertAlmostEqual(speech_sections[0].relative_speaker_change_points[0], 6.944)

        self.assertEqual(len(speech_sections[0].wav_tensor), 8757504)


if __name__ == '__main__':
    unittest.main()