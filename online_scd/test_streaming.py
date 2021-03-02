import unittest
import asyncio
import numpy as np
import torch

from streaming import InputFrameGenerator, AudioStream2MelSpectrogram, StreamingSlidingWindowCmn, StreamingDecoder
import online_scd.trs as trs
import online_scd.data as data
from online_scd.model import SCDModel
from online_scd.trs import Transcritpion

class TestInputFrameGenerator(unittest.TestCase):

    def test_frames(self):
        raw_input1 = np.arange(500)
        raw_input2 = np.arange(500, 1000)
        ifg = InputFrameGenerator(400, 160)
        result = list(ifg.frames(raw_input1)) + list(ifg.frames(raw_input2))
        
        self.assertEqual(result[0].tolist(), raw_input1[0:400].tolist())
        self.assertEqual(result[1].tolist(), raw_input1[160:500].tolist() + raw_input2[0:60].tolist())

    def test_2d(self):
        raw_input1 = np.random.random((100, 2))
        raw_input2 = np.random.random((100, 2))
        ifg = InputFrameGenerator(30, 10)
        result = list(ifg.frames(raw_input1)) + list(ifg.frames(raw_input2))
        np.testing.assert_almost_equal(result[0], raw_input1[0:30, :].tolist())


class TestAudioStream2MelSpectrogram(unittest.TestCase):

    def test_features(self):
        audio = trs.load_wav_file("test/sample_dataset/3321821.wav", 16000)[0: 16000]
        features = data.extract_features(audio, 16000, 40)

        a2s = AudioStream2MelSpectrogram(16000, 40)
        streamed_features = []
        for i in range(0, len(audio), 1000):
            for feature in a2s.process_audio(audio[i: i+1000]):
                 streamed_features.append(feature)

        
        self.assertEqual(len(features), len(streamed_features))

        #breakpoint()
        np.testing.assert_almost_equal(features[-1].tolist(), streamed_features[-1].tolist(), decimal=3)


class TestStreamingSlidingWindowCmn(unittest.TestCase):

    def test_sliding_window_cmn(self):
        cmn = StreamingSlidingWindowCmn(num_feats=2, cmn_window=5)
        input_data = np.random.random((2, 100))
        output_data = np.zeros((2, 100))
        for i in range(input_data.shape[1]):
            output_data[:, i] = cmn.process(input_data[:, i])

        
        np.testing.assert_almost_equal(output_data[:, 9], input_data[:, 9] - input_data[:, 5:10].mean(1))


class TestModel(unittest.TestCase):
    
    def test_decoding(self):
        model = SCDModel.load_from_checkpoint("test/sample_model/checkpoints/epoch=102.ckpt")
        
        transcription = Transcritpion("test/sample_dataset/71_ID117_344945.wav", "test/sample_dataset/71_ID117_344945.trs")
        speech_sections = transcription.get_speech_sections()
        audio = speech_sections[0].wav_tensor[0:16000*100]
        mel_spec = data.extract_features(audio, 16000, model.hparams.num_fbanks).unsqueeze(0)
        mel_spec_length = torch.tensor(mel_spec.shape[-2]).unsqueeze(0)

        output_enc_padded = model._encode_features(mel_spec, mel_spec_length)
        
        logits = model._decode(output_enc_padded,  (mel_spec_length - model.encoder_fov) // model.hparams.detection_period)
        nonstreaming_breaks = logits.log_softmax(dim=-1).squeeze().argmax(1).nonzero(as_tuple=True)[0]

        streaming_model = StreamingDecoder(model)
        streaming_outputs = []
        for i in range(0, len(audio), 1000):
            for output in streaming_model.process_audio(audio[i: i+1000]):
                streaming_outputs.append(output)
        
        streaming_breaks = torch.stack(streaming_outputs).squeeze().argmax(1).nonzero(as_tuple=True)[0] \
            + (model.hparams.label_delay - model.encoder_fov//2) // model.hparams.detection_period
        # Assert that the overlap between streaming and non-streaming is more than 90%
        print("Breaks from non-streaming decoding:", nonstreaming_breaks)
        print("Breaks from streaming decoding:", streaming_breaks)
        self.assertTrue(len(np.intersect1d(nonstreaming_breaks.numpy(), streaming_breaks.numpy())) / len(streaming_breaks) > 0.9)
        self.assertTrue(len(np.intersect1d(nonstreaming_breaks.numpy(), streaming_breaks.numpy())) / len(nonstreaming_breaks) > 0.9)
        

    def test_streaming_with_times(self):
        model = SCDModel.load_from_checkpoint("test/sample_model/checkpoints/epoch=102.ckpt")
        
        transcription = Transcritpion("test/sample_dataset/71_ID117_344945.wav", "test/sample_dataset/71_ID117_344945.trs")
        speech_sections = transcription.get_speech_sections()
        audio = speech_sections[0].wav_tensor
        print("True speaker change points: ", speech_sections[0].relative_speaker_change_points)
        streaming_decoder = StreamingDecoder(model)
        streaming_outputs = []
        for i in range(0, len(audio), 1000):
            for time in streaming_decoder.find_speaker_change_times(audio[i: i+1000]):
                print("Found speaker change point: ", time)


if __name__ == '__main__':
    unittest.main()