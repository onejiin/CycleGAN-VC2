import argparse
import os
import numpy as np

from model import CycleGAN
from preprocess import *

def conversion(model_dir, model_name, data_dir, conversion_direction, output_dir, pc, generation_model):

    num_features = 32
    sampling_rate = 16000
    frame_period = 5.0

    model = CycleGAN(num_features = num_features, mode = 'test', gen_model=generation_model)

    model.load(filepath = os.path.join(model_dir, model_name))

    mcep_normalization_params = np.load(os.path.join(model_dir, 'mcep_normalization.npz'))
    mcep_mean_A = mcep_normalization_params['mean_A']
    mcep_std_A = mcep_normalization_params['std_A']
    mcep_mean_B = mcep_normalization_params['mean_B']
    mcep_std_B = mcep_normalization_params['std_B']

    logf0s_normalization_params = np.load(os.path.join(model_dir, 'logf0s_normalization.npz'))
    logf0s_mean_A = logf0s_normalization_params['mean_A']
    logf0s_std_A = logf0s_normalization_params['std_A']
    logf0s_mean_B = logf0s_normalization_params['mean_B']
    logf0s_std_B = logf0s_normalization_params['std_B']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(data_dir):

        filepath = os.path.join(data_dir, file)
        wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
        # wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sampling_rate, dim = num_features)
        coded_sp_transposed = coded_sp.T

        frame_size = 128
        if conversion_direction == 'A2B':
            # pitch
            print("AtoB")
            if pc == True:
                print("pitch convert")
                f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A,
                 mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
            else:
                print("pitch same")
                f0_converted = f0

            # normalization A Domain
            coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A

            # padding
            remain, padd = frame_size - coded_sp_norm.shape[1] % frame_size, False
            if coded_sp_norm.shape[1] % frame_size != 0:
                coded_sp_norm = np.concatenate((coded_sp_norm, np.zeros((32, remain))), axis=1)
                padd = True

            # inference for segmentation
            coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm[:, 0:frame_size]]), direction=conversion_direction)[0]
            for i in range(1, coded_sp_norm.shape[1] // frame_size):
                ccat = model.test(inputs=np.array([coded_sp_norm[:, i * frame_size:(i + 1) * frame_size]]),
                                  direction=conversion_direction)[0]
                coded_sp_converted_norm = np.concatenate((coded_sp_converted_norm, ccat), axis=1)

            if padd == True:
                coded_sp_converted_norm = coded_sp_converted_norm[:,:-remain]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B
        else:
            print("BtoA")
            if pc == True:
                print("pitch convert")
                f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A,
                mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
            else:
                f0_converted = f0

            # normalization B Domain
            coded_sp_norm = (coded_sp_transposed - mcep_mean_B) / mcep_std_B

            # padding
            remain, padd = frame_size - coded_sp_norm.shape[1] % frame_size, False
            if coded_sp_norm.shape[1] % frame_size != 0:
                coded_sp_norm = np.concatenate((coded_sp_norm, np.zeros((32, remain))), axis=1)
                padd = True

            # inference for segmentation
            coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm[:, 0:frame_size]]), direction=conversion_direction)[0]
            for i in range(1, coded_sp_norm.shape[1] // frame_size):
                ccat = model.test(inputs=np.array([coded_sp_norm[:, i * frame_size:(i + 1) * frame_size]]),
                                  direction=conversion_direction)[0]
                coded_sp_converted_norm = np.concatenate((coded_sp_converted_norm, ccat), axis=1)

            if padd == True:
                coded_sp_converted_norm = coded_sp_converted_norm[:,:-remain]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_A + mcep_mean_A

        # output translation value processing
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)

        # World vocoder synthesis
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sampling_rate, frame_period = frame_period)
        librosa.output.write_wav(os.path.join(output_dir, os.path.basename(file)), wav_transformed, sampling_rate)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained CycleGAN model.')

    model_dir_default = './model/sf1_tm1'
    model_name_default = 'sf1_tm1.ckpt'
    data_dir_default = './data/evaluation_all/SF1'
    conversion_direction_default = 'A2B'
    output_dir_default = './converted_voices'
    pc_default = True
    generation_model_default='CycleGAN-VC2'

    parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'Filename for the pre-trained model.', default = model_name_default)
    parser.add_argument('--data_dir', type = str, help = 'Directory for the voices for conversion.', default = data_dir_default)
    parser.add_argument('--conversion_direction', type=str,
                        help='Conversion direction for CycleGAN. A2B or B2A. The first object in the model file name is A, and the second object in the model file name is B.',
                        default=conversion_direction_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the converted voices.', default = output_dir_default)
    parser.add_argument('--pc', type=bool, help='True: using pitch conversion in DomainB',
                        default=pc_default)
    parser.add_argument('--generation_model', type=str, help='generator_gatedcnn / generator_gatedcnn_SAGAN',
                        default=generation_model_default)

    argv = parser.parse_args()

    model_dir = argv.model_dir
    model_name = argv.model_name
    data_dir = argv.data_dir
    conversion_direction = argv.conversion_direction
    output_dir = argv.output_dir
    pc = argv.pc
    plot_ox = argv.plot_ox
    generation_model=argv.generation_model

    # Conversion coder
    conversion(model_dir = model_dir, model_name = model_name, data_dir = data_dir, conversion_direction = conversion_direction, output_dir = output_dir, pc=pc, generation_model=generation_model)
