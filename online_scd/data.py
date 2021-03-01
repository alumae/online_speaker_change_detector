import torch
import online_scd.trs as trs
import tqdm
import random
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from argparse import ArgumentParser
import logging
import audiomentations
from scipy.io import wavfile
import librosa

EPSILON = torch.tensor(torch.finfo(torch.float).eps)

def dump_for_debug(audio_tensor, sample_rate, chunk_frame_labels, idx, dir="tmp"):
    tmp_audio_tensor = audio_tensor.copy()
    marker_tensor = torch.rand(int(0.1 * sample_rate))
    print(f"Chunk {idx}: relative change points: " + ", ".join([str(x/100.) for x in chunk_frame_labels.nonzero()]))

    for frame_change_points in chunk_frame_labels.nonzero():
        change_point_in_samples = int(frame_change_points / 100.0 * sample_rate)

        tmp_audio_tensor[change_point_in_samples : change_point_in_samples+len(marker_tensor)] = marker_tensor
    wavfile.write(f"{dir}/chunk_{idx}.wav", sample_rate, tmp_audio_tensor)


def extract_features(audio, sample_rate, num_fbanks):        
    assert(len(audio.shape) == 1)
    feat = librosa.feature.melspectrogram(y=audio, sr=sample_rate, 
        center=False,
        n_fft=int(2.5*sample_rate/100.0), hop_length=sample_rate//100,
        fmin=40, fmax=sample_rate//2-400, n_mels=num_fbanks)
    feat = torch.from_numpy(feat.T)
    feat = torch.max(feat, EPSILON).log()
    feat = torchaudio.functional.sliding_window_cmn(feat)
    return feat

class SCDDataset(torch.utils.data.Dataset):
    def __init__(self, dir, extract_chunks=True, sample_rate=16000, num_fbanks=40, label_delay=100, no_augment=False, **kwargs):
        self.extract_chunks = extract_chunks
        self.min_length = kwargs["min_chunk_length"]
        self.max_length = kwargs["max_chunk_length"]
        self.sample_rate = sample_rate
        self.num_fbanks = num_fbanks
        self.label_delay = label_delay
        reco2wav = {}
        reco2trs = {}
        with open(f"{dir}/wav.scp") as f:
            for l in f:
                ss = l.split()
                reco2wav[ss[0]] = ss[1]

        with open(f"{dir}/reco2trs.scp") as f:
            for l in f:
                ss = l.split()
                reco2trs[ss[0]] = ss[1]

        self.sections = []
        # Sections are of different length; 
        # we add a section section_length/avg_chunk_len + 1 number of times
        self.index2section = []

        avg_chunk_len = self.max_length - self.min_length

        for reco in tqdm.tqdm(reco2trs.keys(), desc=f"Loading transcriptions and audios for {dir}"):
            try:
                transcription = trs.Transcritpion(reco2wav[reco], reco2trs[reco])
                for section in transcription.get_speech_sections():
                    self.sections.append(section)
                    section_length = section.wav_tensor.shape[0] / sample_rate
                    if extract_chunks:
                        self.index2section.extend([len(self.sections) - 1]  * int(section_length // avg_chunk_len + 1))
                    else:
                        self.index2section.append(len(self.sections) - 1)
            except:
                logging.warn(f"Cannot load transcription/audio for {reco}",  exc_info=True)

        self.augment = None
        if not no_augment:
            augmentations = []
            if kwargs["rir_dir"] != "":
                augmentations.append(audiomentations.AddImpulseResponse(ir_path=kwargs["rir_dir"], p=0.3, lru_cache_size=1024))
            if kwargs["noise_dir"] != "":
                augmentations.append(audiomentations.AddBackgroundNoise(sounds_path=kwargs["noise_dir"], p=0.3, lru_cache_size=1024))
            if kwargs["short_noise_dir"] != "":
                augmentations.append(audiomentations.AddShortNoises(sounds_path=kwargs["short_noise_dir"], p=0.3, lru_cache_size=1024))
            if len(augmentations) > 0:
                self.augment = audiomentations.Compose(augmentations)


    def __len__(self):
        return len(self.index2section)

    def __getitem__(self, idx):
        return self.sections[self.index2section[idx]]



    def collater(self, sections):
        mel_spec = []
        mel_spec_lengths = torch.zeros(len(sections),  dtype=torch.int32)
        labels = []
        if self.extract_chunks:
            chunk_length = random.uniform(self.min_length, self.max_length)
            chunk_length_in_samples = int(chunk_length * self.sample_rate)
            for i, section in enumerate(sections):
                audio_tensor = section.wav_tensor
                # TODO: speed perturbation                
                start_pos = random.randint(0, max(0, len(audio_tensor) - chunk_length_in_samples))
                current_chunk_length = min(chunk_length_in_samples, len(audio_tensor))
                chunk = audio_tensor[start_pos:start_pos+current_chunk_length]
                if self.augment is not None:
                    chunk = self.augment(samples=chunk, sample_rate=self.sample_rate)[0:current_chunk_length]

                current_mel_spec = extract_features(chunk, self.sample_rate, self.num_fbanks)    

                mel_spec.append(current_mel_spec)
                mel_spec_lengths[i] = current_mel_spec.shape[0]
                chunk_frame_labels = torch.zeros(mel_spec_lengths[i], dtype=torch.int64)
                for speaker_change_point in section.relative_speaker_change_points:
                    shifted_change_point_in_frames = int(speaker_change_point * self.sample_rate - start_pos) * 100 // self.sample_rate + self.label_delay
                    if shifted_change_point_in_frames > 0 and shifted_change_point_in_frames < mel_spec_lengths[i]:
                        chunk_frame_labels[shifted_change_point_in_frames] = 1
                labels.append(chunk_frame_labels)
                #dump_for_debug(chunk, self.sample_rate, chunk_frame_labels, i)
        else:
            for i, section in enumerate(sections):
                audio_tensor = section.wav_tensor
                if self.augment is not None:
                    audio_tensor = self.augment(samples=audio_tensor, sample_rate=self.sample_rate)[0:len(audio_tensor)]

                current_mel_spec = extract_features(audio_tensor, self.sample_rate, self.num_fbanks)    
                mel_spec.append(current_mel_spec)
                mel_spec_lengths[i] = current_mel_spec.shape[0]
                chunk_frame_labels = torch.zeros(mel_spec_lengths[i], dtype=torch.int64)
                for speaker_change_point in section.relative_speaker_change_points:
                    shifted_change_point_in_frames = int(speaker_change_point * self.sample_rate) * 100 // self.sample_rate + self.label_delay
                    if shifted_change_point_in_frames > 0 and shifted_change_point_in_frames < mel_spec_lengths[i]:
                        chunk_frame_labels[shifted_change_point_in_frames] = 1
                labels.append(chunk_frame_labels)
        
        batch = {
            "mel_spec": pad_sequence(mel_spec, batch_first=True),
            "mel_spec_length": mel_spec_lengths,
            "label": pad_sequence(labels, batch_first=True, padding_value=0)
        }
        return batch

    @staticmethod
    def add_data_specific_args(parent_parser, root_dir):  # pragma: no cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--rir-dir', default="", type=str)
        parser.add_argument('--noise-dir', default="", type=str)
        parser.add_argument('--short-noise-dir', default="", type=str)
        parser.add_argument('--min-chunk-length', default=10.0, type=float)
        parser.add_argument('--max-chunk-length', default=30.0, type=float)
        return parser


