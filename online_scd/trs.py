import sys
from collections import namedtuple
from lxml import objectify
import torch

import numpy as np
import ctypes
import multiprocessing as mp
from online_scd.utils import load_wav_file

SpeechSectionInfo = namedtuple("SpeechSectionInfo",
    ['start_time', 'end_time', 'speaker_change_points'])

SpeechSection = namedtuple("SpeechSection",
    ['wav_tensor', 'relative_speaker_change_points'])


class Transcritpion():
    def __init__(self, wav_filename, trs_filename, sample_rate=16000):
        self.speech_section_infos = parse_trs(trs_filename) 
        wav_tensor_tmp = load_wav_file(wav_filename, sample_rate)
        shared_array_base = mp.RawArray(ctypes.c_float, len(wav_tensor_tmp))
        self.wav_tensor = np.frombuffer(shared_array_base, dtype=np.float32).reshape(len(wav_tensor_tmp))
        self.wav_tensor[:] = wav_tensor_tmp
        self.sample_rate = sample_rate

    def get_speech_sections(self):
        return [SpeechSection(self.wav_tensor[int(i.start_time*self.sample_rate): int(i.end_time*self.sample_rate)], i.speaker_change_points - i.start_time)
                for i in self.speech_section_infos]


def parse_trs(filename):
    with open(filename, 'rb') as f:
        tree = objectify.parse(f)
        sections = tree.getroot()["Episode"]["Section"]
        speech_sections = []
        for section in sections:
            if section.attrib['type'] == 'report':
                start_time = float(section.attrib['startTime'])
                end_time = float(section.attrib['endTime'])
                speaker_change_points = []
                last_speaker = None
                turns = section["Turn"]
                for turn in turns:
                    if last_speaker is not None and "speaker" in turn.attrib and turn.attrib["speaker"] != last_speaker:
                        speaker_change_points.append(float(turn.attrib["startTime"]))
                    if "speaker" in turn.attrib:
                        last_speaker = turn.attrib["speaker"]
                speech_sections.append(SpeechSectionInfo(
                    start_time=start_time, end_time=end_time, speaker_change_points=np.array(speaker_change_points)))
        return speech_sections
