# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
from preprocess import world_decompose, world_encode_spectral_envelop
import librosa


def world_encode_data_toSave_spec(num_mcep, hdf5_dir, wav_dir, sr, frame_period=5.0, coded_dim24=24, coded_dim36=36):
    # 1. 32dim MCEP         [32  x frame]
    # 2. 512dim cheaptrick  [512 x frame]
    # 3. 512dim spectrogram [512 x frame]
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps24 = list()
    coded_sps36 = list()
    coded_sps32 = list()
    spectrograms = list()

    def calc_spec_wav(wav, f0):

        sr = 16000
        duration = 0.005
        hop_size = int(sr * duration) # 16000 * 0.005 = 80 sample
        spectrograms = list()

        for i in range(wav.shape[0] // (hop_size) + 1):
            start = i * hop_size
            if f0[i] == 0:
                segment_wav = wav[start : start+1024]
            else:
                segment_wav = wav[start : start+int(3 * 1 / f0[i] * 16000)]
            fft_size = 1024
            # D = np.abs(librosa.stft(segment_wav, n_fft=fft_size, hop_length=segment_wav.shape[0]*2))
            if segment_wav.shape[0] == 0:
                D = np.zeros((513))
            else:
                D = np.abs(librosa.stft(segment_wav, n_fft=fft_size, hop_length=2048))[:, 0]
            magnitude = D#[:-1]
            spectrograms.append(magnitude)

        return spectrograms

    for file in os.listdir(wav_dir):
        print("----")
        print(file)
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)

        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sr, frame_period=frame_period)
        spectrogram = calc_spec_wav(wav, f0)

        coded_sp24 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=coded_dim24)
        coded_sp36 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=coded_dim36)
        coded_sp32 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=32)
        spectrogram = np.array(spectrogram)

        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps24.append(coded_sp24)
        coded_sps36.append(coded_sp36)
        coded_sps32.append(coded_sp32)
        spectrograms.append(spectrogram)

        # file write
        item = {"f0": f0, "timeaxe": timeaxis, "ap": ap, "sp": sp, "coded24": coded_sp24, "coded36": coded_sp36,
                "coded32": coded_sp32, "spectrogram": spectrogram}
        hdf5_file_name = os.path.join(hdf5_dir, os.path.splitext(file)[0]+".h5")
        hdf5_write(hdf5_file_name, item)

    assert num_mcep == 24 or num_mcep == 36 or num_mcep == 32, "spectral envelop dimension misatch"
    if num_mcep == 24:
        coded_sps = coded_sps24
    elif num_mcep == 36:
        coded_sps = coded_sps36
    elif num_mcep == 32:
        coded_sps = coded_sps32

    return f0s, timeaxes, sps, aps, coded_sps, spectrograms


def world_encode_data_toSave(num_mcep, hdf5_dir, wav_dir, sr, frame_period=5.0, coded_dim24=24, coded_dim36=36):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps24 = list()
    coded_sps36 = list()
    coded_sps32 = list()

    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)

        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sr, frame_period=frame_period)
        coded_sp24 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=coded_dim24)
        coded_sp36 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=coded_dim36)
        coded_sp32 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=32)

        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps24.append(coded_sp24)
        coded_sps36.append(coded_sp36)
        coded_sps32.append(coded_sp32)

        # file write
        item = {"f0": f0, "timeaxe": timeaxis, "ap": ap, "sp": sp, "coded24": coded_sp24, "coded36": coded_sp36, "coded32": coded_sp32}
        hdf5_file_name = os.path.join(hdf5_dir, os.path.splitext(file)[0]+".h5")
        hdf5_write(hdf5_file_name, item)

    assert num_mcep == 24 or num_mcep == 36 or num_mcep == 32, "spectral envelop dimension misatch"
    if num_mcep == 24:
        coded_sps = coded_sps24
    elif num_mcep == 36:
        coded_sps = coded_sps36
    elif num_mcep == 32:
        coded_sps = coded_sps32

    return f0s, timeaxes, sps, aps, coded_sps


def world_encode_data_toLoad(num_mcep, hdf5_dir):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps24 = list()
    coded_sps36 = list()
    coded_sps32 = list()

    for file in os.listdir(hdf5_dir):
        file_path = os.path.join(hdf5_dir, file)
        file = h5py.File(file_path, 'r')

        f0 = file["f0"].value
        timeaxe = file["timeaxe"].value
        sp = file["ap"].value
        ap = file["sp"].value
        coded_sp24 = file["coded24"].value
        coded_sp36 = file["coded36"].value
        coded_sp32 = file["coded32"].value

        f0s.append(f0)
        timeaxes.append(timeaxe)
        sps.append(sp)
        aps.append(ap)
        coded_sps24.append(coded_sp24)
        coded_sps36.append(coded_sp36)
        coded_sps32.append(coded_sp32)

    assert num_mcep == 24 or num_mcep == 36 or num_mcep == 32, "spectral envelop dimension misatch"
    if num_mcep == 24:
        coded_sps = coded_sps24
    elif num_mcep == 36:
        coded_sps = coded_sps36
    elif num_mcep == 32:
        coded_sps = coded_sps32

    return f0s, timeaxes, sps, aps, coded_sps

def world_encode_data_toLoad_spec(num_mcep, hdf5_dir):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps24 = list()
    coded_sps36 = list()
    coded_sps32 = list()
    spectrograms = list()

    for file in os.listdir(hdf5_dir):
        file_path = os.path.join(hdf5_dir, file)
        file = h5py.File(file_path, 'r')

        f0 = file["f0"].value
        timeaxe = file["timeaxe"].value
        sp = file["ap"].value
        ap = file["sp"].value
        coded_sp24 = file["coded24"].value
        coded_sp36 = file["coded36"].value
        coded_sp32 = file["coded32"].value
        spectrogram = file["spectrogram"].value

        f0s.append(f0)
        timeaxes.append(timeaxe)
        sps.append(sp)
        aps.append(ap)
        coded_sps24.append(coded_sp24)
        coded_sps36.append(coded_sp36)
        coded_sps32.append(coded_sp32)
        spectrograms.append(spectrogram)

    assert num_mcep == 24 or num_mcep == 36 or num_mcep == 32, "spectral envelop dimension misatch"
    if num_mcep == 24:
        coded_sps = coded_sps24
    elif num_mcep == 36:
        coded_sps = coded_sps36
    elif num_mcep == 32:
        coded_sps = coded_sps32

    return f0s, timeaxes, sps, aps, coded_sps, spectrograms

def hdf5_write(file_name, item):
    dset = h5py.File(file_name, 'w')

    dset['f0'] = item["f0"]
    dset['timeaxe'] = item["timeaxe"]
    dset['ap'] = item["ap"]
    dset['sp'] = item["sp"]
    dset['coded24'] = item["coded24"]
    dset['coded36'] = item["coded36"]
    dset['coded32'] = item["coded32"]
    dset['spectrogram'] = item["spectrogram"]
    dset.close()


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hdf5 DB write')

    wav_file_default = '/root/onejin/S2SCycleGAN/data/debug/ma/arctic_a0010.wav'
    hdf5_file_default = '/root/onejin/S2SCycleGAN/data/debug_hdf5/ma/arctic_a0010.h5'
    parser.add_argument('--wav_file', type=str, help='Directory for A.', default=wav_file_default)
    parser.add_argument('--hdf_file', type=str, help='Directory for B.', default=hdf5_file_default)

    argv = parser.parse_args()
    wav_file = argv.wav_file
    hdf_file = argv.hdf_file

    sr = 16000
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)

    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sr, frame_period=5.0)
    def calc_spec_wav(wav, f0):

        sr = 16000
        duration = 0.005
        hop_size = int(sr * duration) # 16000 * 0.005 = 80 sample
        spectrograms = list()

        for i in range(wav.shape[0] // (hop_size) + 1):
            start = i * hop_size
            if f0[i] == 0:
                segment_wav = wav[start : start+1024]
            else:
                segment_wav = wav[start : start+int(3 * 1 / f0[i] * 16000)]
            fft_size = 1024
            # D = np.abs(librosa.stft(segment_wav, n_fft=fft_size, hop_length=segment_wav.shape[0]*2))
            if segment_wav.shape[0] == 0:
                D = np.zeros((513))
            else:
                D = np.abs(librosa.stft(segment_wav, n_fft=fft_size, hop_length=2048))[:, 0]
            magnitude = D#[:-1]
            spectrograms.append(magnitude)

        return spectrograms

    spectrogram = calc_spec_wav(wav, f0)

    coded_sp24 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=24)
    coded_sp36 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=36)
    coded_sp32 = world_encode_spectral_envelop(sp=sp, fs=sr, dim=32)
    spectrogram = np.array(spectrogram)


    # file write
    item = {"f0": f0, "timeaxe": timeaxis, "ap": ap, "sp": sp, "coded24": coded_sp24, "coded36": coded_sp36,
            "coded32": coded_sp32, "spectrogram": spectrogram}
    hdf5_file_name = hdf_file
    hdf5_write(hdf5_file_name, item)

