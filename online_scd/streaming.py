import torch

import functools
import numpy as np
from asyncio import Queue
import librosa

import online_scd.data as data

class InputFrameGenerator(object):

    def __init__(self, blocksize, stepsize):
        self.blocksize = blocksize
        self.stepsize = stepsize
        self.buffer = None

    def frames(self, frames):
        if self.buffer is not None:
            stack = np.concatenate([self.buffer, frames])
        else:
            stack = frames.copy()

        stack_length = len(stack)

        nb_frames = (
            stack_length - self.blocksize + self.stepsize) // self.stepsize
        nb_frames = max(nb_frames, 0)
        frames_length = nb_frames * self.stepsize + \
            self.blocksize - self.stepsize
        last_block_size = stack_length - frames_length

        self.buffer = stack[int(nb_frames * self.stepsize):]

        for index in range(0, int(nb_frames * self.stepsize), int(self.stepsize)):
            yield stack[index:index + self.blocksize]


class StreamingSlidingWindowCmn:
    def __init__(self, num_feats, cmn_window=600):
        self.cmn_window = cmn_window
        self.rolling_position = 0
        self.rolling_buffer = np.zeros((num_feats, cmn_window))
        self.buffer_length = 0

    def process(self, frame):
        self.rolling_buffer[:, self.rolling_position] = frame
        self.rolling_position = (self.rolling_position + 1) % self.cmn_window        
        self.buffer_length = min(self.buffer_length + 1, self.cmn_window)
        return frame - self.rolling_buffer[:, 0:self.buffer_length].mean(1)


class AudioStream2MelSpectrogram:
    def __init__(self, sample_rate=16000, num_fbanks=40, cmn_window=600):
        self.sample_rate = sample_rate
        self.num_fbanks = num_fbanks
        self.input_frame_generator = InputFrameGenerator(400, 160)
        self.result_queue = Queue()
        self.cmn = StreamingSlidingWindowCmn(num_fbanks, cmn_window)

    def process_audio(self, audio):
        for frames in self.input_frame_generator.frames(audio):
            single_feat = librosa.feature.melspectrogram(frames, sr=self.sample_rate,
                                            center=False,
                                            n_fft=int(2.5*self.sample_rate/100.0), hop_length=self.sample_rate//100,
                                            fmin=40, fmax=self.sample_rate//2-400, n_mels=self.num_fbanks)
            
            single_feat = single_feat[:, 0]
            single_feat = np.log(np.clip(single_feat, data.EPSILON.numpy(), None))
            single_feat = self.cmn.process(single_feat)
            
            yield single_feat
        

class StreamingDecoder:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.audio2mel = AudioStream2MelSpectrogram(16000, model.hparams.num_fbanks)
        self.mels_to_conv_input = InputFrameGenerator(model.encoder_fov, model.hparams.detection_period)
        self.hidden_state = None
        self.frame_counter = 0
        self.starting_buffer = [torch.tensor([1.0, 0.0])] * (model.encoder_fov // model.hparams.detection_period - 2) 

    def process_audio(self, audio):        
        for feature in self.audio2mel.process_audio(audio):            
            for x in self.mels_to_conv_input.frames(feature.reshape(1, self.model.hparams.num_fbanks)):
                x = torch.from_numpy(x).permute(1, 0).unsqueeze(0).float()
                x = self.model.encode_windowed_features(x)
                y, self.hidden_state = self.model.decode_single_timestep(x, self.hidden_state)
                # first yield the pseudo-states for the frames that were skipped due to no padding
                if len(self.starting_buffer) > 0:
                    yield self.starting_buffer.pop()

                yield y.squeeze()

    def find_speaker_change_times(self, audio, threshold=0.5):        
        for y in self.process_audio(audio):

            if y.exp()[1] > threshold:
                change_time = self.frame_counter / 100 - 1.0
                if change_time > 0:
                    yield change_time

            self.frame_counter += 10



