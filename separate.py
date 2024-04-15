import os
import sys
import torch
import warnings
import pdb
import argparse
import hashlib
import math
import librosa
import importlib
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from uvr5_pack.lib_v5 import spec_utils
from uvr5_pack.utils import _get_name_params, inference
from uvr5_pack.lib_v5.model_param_init import ModelParameters

def separate_audio(audio_path, save_path):
    class _audio_pre_():
        def __init__(self, model_path, device, is_half):
            self.model_path = model_path
            self.device = device
            self.data = {
                # Processing Options
                'postprocess': False,
                'tta': False,
                # Constants
                'window_size': 512,
                'agg': 10,
                'high_end_process': 'mirroring',
            }
            nn_arch_sizes = [
                31191, # default
                33966, 61968, 123821, 123812, 537238 # custom
            ]
            self.nn_architecture = list('{}KB'.format(s) for s in nn_arch_sizes)
            model_size = math.ceil(os.stat(model_path).st_size / 1024)
            nn_architecture = '{}KB'.format(min(nn_arch_sizes, key=lambda x: abs(x-model_size)))
            nets = importlib.import_module('uvr5_pack.lib_v5.nets' + f'_{nn_architecture}'.replace('_{}KB'.format(nn_arch_sizes[0]), ''), package=None)
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
            param_name, model_params_d = _get_name_params(model_path, model_hash)

            mp = ModelParameters(model_params_d)
            model = nets.CascadedASPPNet(mp.param['bins'] * 2)
            cpk = torch.load(model_path, map_location='cpu')  
            model.load_state_dict(cpk)
            model.eval()
            if is_half:
                model = model.half().to(device)
            else:
                model = model.to(device)

            self.mp = mp
            self.model = model

        def _path_audio_(self, music_file, ins_root=None, vocal_root=None):
            if ins_root is None and vocal_root is None:
                return "No save root."
            name = os.path.basename(music_file)
            if ins_root is not None:
                os.makedirs(ins_root, exist_ok=True)
            if vocal_root is not None:
                os.makedirs(vocal_root, exist_ok=True)
            X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
            bands_n = len(self.mp.param['band'])
            for d in range(bands_n, 0, -1): 
                bp = self.mp.param['band'][d]
                if d == bands_n: 
                    X_wave[d], _ = librosa.core.load(
                        music_file, bp['sr'], False, dtype=np.float32, res_type=bp['res_type'])
                    if X_wave[d].ndim == 1:
                        X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
                else: 
                    X_wave[d] = librosa.core.resample(X_wave[d+1], self.mp.param['band'][d+1]['sr'], bp['sr'], res_type=bp['res_type'])
                X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], self.mp.param['mid_side'], self.mp.param['mid_side_b2'], self.mp.param['reverse'])
                if d == bands_n and self.data['high_end_process'] != 'none':
                    input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                    input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]
            X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
            aggresive_set = float(self.data['agg']/100)
            aggressiveness = {'value': aggresive_set, 'split_bin': self.mp.param['band'][1]['crop_stop']}
            with torch.no_grad():
                pred, X_mag, X_phase = inference(X_spec_m, self.device, self.model, aggressiveness, self.data)
            if self.data['postprocess']:
                pred_inv = np.clip(X_mag - pred, 0, np.inf)
                pred = spec_utils.mask_silence(pred, pred_inv)
            y_spec_m = pred * X_phase
            v_spec_m = X_spec_m - y_spec_m
            if ins_root is not None:
                if self.data['high_end_process'].startswith('mirroring'):
                    input_high_end_ = spec_utils.mirroring(self.data['high_end_process'], y_spec_m, input_high_end, self.mp)
                    wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp, input_high_end_h, input_high_end_)
                else:
                    wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
                print('%s instruments done' % name)
                wavfile.write(os.path.join(ins_root, 'instrument_{}.wav'.format(name)), self.mp.param['sr'], (np.array(wav_instrument)*32768).astype("int16"))  
            if vocal_root is not None:
                if self.data['high_end_process'].startswith('mirroring'):
                    input_high_end_ = spec_utils.mirroring(self.data['high_end_process'],  v_spec_m, input_high_end, self.mp)
                    wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp, input_high_end_h, input_high_end_)
                else:
                    wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
                print('%s vocals done' % name)
                wavfile.write(os.path.join(vocal_root, 'vocal_{}.wav'.format(name)), self.mp.param['sr'], (np.array(wav_vocals)*32768).astype("int16"))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_half = True
    model_path = 'R:/Big Programming Projects/Python/SoundsLike/uvr/uvr5_weights/2_HP-UVR.pth'

    pre_fun = _audio_pre_(model_path=model_path, device=device, is_half=True)
    pre_fun._path_audio_(audio_path, save_path, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file')
    parser.add_argument('save_path', type=str, help='Path to save the processed audio files')
    args = parser.parse_args()

    separate_audio(args.audio_path, args.save_path)