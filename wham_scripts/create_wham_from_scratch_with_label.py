import os
import numpy as np
import soundfile as sf
import pandas as pd
from constants import SAMPLERATE
import argparse
import json
from utils_with_label import read_scaled_wav, quantize, fix_length, create_wham_mixes, append_or_truncate


FILELIST_STUB = os.path.join('data', 'mix_2_spk_filenames_{}.csv')

SINGLE_DIR = 'mix_single'
BOTH_DIR = 'mix_both'
CLEAN_DIR = 'mix_clean'
S1_DIR = 's1'
S2_DIR = 's2'
NOISE_DIR = 'noise'
MIX_LABEL = 'mix_label'

def create_wham(wsj_root, wham_noise_path, output_root, json_root, wsjmix_16k_root=None, wsjmix_8k_root=None):
    
    scaling_npz_stub = os.path.join(wham_noise_path, 'metadata', 'scaling_{}.npz')
    if wsj_root is not None:

        from_scratch = True
    else:
        from_scratch = False

    for splt in ['tr', 'cv', 'tt']:

        wsjmix_path = FILELIST_STUB.format(splt)
        wsjmix_df = pd.read_csv(wsjmix_path)

        scaling_npz_path = scaling_npz_stub.format(splt)
        scaling_npz = np.load(scaling_npz_path, allow_pickle=True)

        noise_path = os.path.join(wham_noise_path, splt)
        #for sr_dir in ['16k', '8k']:
        for sr_dir in ['8k']:
            wav_dir = 'wav' + sr_dir
            if sr_dir == '8k':
                sr = 8000
                downsample = True
                wsjmix_path = wsjmix_8k_root
            else:
                sr = SAMPLERATE
                downsample = False
                wsjmix_path = wsjmix_16k_root

            for datalen_dir in ['max', 'min']:  
                print('{} {} dataset, {} split'.format(sr_dir, datalen_dir, splt))
                json_path = os.path.join(json_root, wav_dir, datalen_dir, splt)
                os.makedirs(json_path, exist_ok=True)
                output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                os.makedirs(os.path.join(output_path, CLEAN_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, SINGLE_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, BOTH_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, S1_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, S2_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, NOISE_DIR), exist_ok=True)
                os.makedirs(os.path.join(output_path, MIX_LABEL), exist_ok=True)

                wsjmix_key = 'scaling_wsjmix_{}_{}'.format(sr_dir, datalen_dir)
                wham_speech_key = 'scaling_wham_speech_{}_{}'.format(sr_dir, datalen_dir)
                wham_noise_key = 'scaling_wham_noise_{}_{}'.format(sr_dir, datalen_dir)

                utt_ids = scaling_npz['utterance_id']
                start_samp_16k = scaling_npz['speech_start_sample_16k']
                scaling_noise = scaling_npz[wham_noise_key]

                scaling_speech = scaling_npz[wham_speech_key]

                json_data = []
                if from_scratch:
                    scaling_wsjmix = scaling_npz[wsjmix_key]

                for i_utt, output_name in enumerate(utt_ids):
                    if from_scratch:
                        utt_row = wsjmix_df[wsjmix_df['output_filename'] == output_name]
                        s1_path = os.path.join(wsj_root, utt_row['s1_path'].iloc[0])
                        s2_path = os.path.join(wsj_root, utt_row['s2_path'].iloc[0])
                        spk_info = []
                        spk_info.append(s1_path.split("/")[-2])
                        spk_info.append(s2_path.split("/")[-2])
                        #####
                        s1_label_path = s1_path[:len(s1_path)-4] + '_time_label.npy'
                        s2_label_path = s2_path[:len(s2_path)-4] + '_time_label.npy'
                        #####
                        s1 = read_scaled_wav(s1_path, scaling_wsjmix[i_utt][0], downsample)
                        s1 = quantize(s1) * scaling_speech[i_utt]
                        s2 = read_scaled_wav(s2_path, scaling_wsjmix[i_utt][1], downsample)
                        s2 = quantize(s2) * scaling_speech[i_utt]
                        #####
                        s1_time_label = np.load(s1_label_path)
                        s2_time_label = np.load(s2_label_path)
                        #####
                        s1_samples, s2_samples, mixture_timestamps = fix_length(s1, s2, s1_time_label, s2_time_label, datalen_dir)
                    else:
                        
                        wsj_path = os.path.join(wsjmix_path, datalen_dir, splt)
                        s1_path = os.path.join(wsj_path, S1_DIR, output_name)
                        s1_samples = read_scaled_wav(s1_path, scaling_speech[i_utt])
                        s2_path = os.path.join(wsj_path, S2_DIR, output_name)
                        s2_samples = read_scaled_wav(s2_path, scaling_speech[i_utt])
                        
                    noise_samples = read_scaled_wav(os.path.join(noise_path, output_name), scaling_noise[i_utt],
                                                    downsample_8K=downsample)
                    s1_samples, s2_samples, noise_samples, Mixture= append_or_truncate(s1_samples, s2_samples, noise_samples, mixture_timestamps,
                                                                               datalen_dir, start_samp_16k[i_utt], downsample)

                    mix_clean, mix_single, mix_both = create_wham_mixes(s1_samples, s2_samples, noise_samples)
                    # write audio
                    samps = [mix_clean, mix_single, mix_both, s1_samples, s2_samples, noise_samples]
                    dirs = [CLEAN_DIR, SINGLE_DIR, BOTH_DIR, S1_DIR, S2_DIR, NOISE_DIR]
                    single = []
                    for dir, samp in zip(dirs, samps):
                        dic = {}
                        dic["Type"] = dir
                        #print("Type:",dir)
                        dic["Src"] = os.path.join(output_path, dir, output_name)
                        dic["Sample"] = len(samp)
                        if dir in ["s1","s2"]:
                            dic["spkID"] = spk_info[int(dir[-1])-1]
                        # print(" src:",os.path.join(output_path, dir, output_name))
                        sf.write(os.path.join(output_path, dir, output_name), samp,
                                 sr, subtype='FLOAT')
                        # print(" sample:",len(samp))
                        # if dir in ["s1","s2"]:
                        #     print(" spkID:",spk_info[int(dir[-1])-1])
                        single.append(dic)
                    np.save(os.path.join(output_path, MIX_LABEL, output_name[:-4]+'.npy'),Mixture)
                    #print(os.path.join(output_path, MIX_LABEL, output_name[:-4]+'.npy'))
                    #print(Mixture.shape)
                    json_data.append(single)
                    if (i_utt + 1) % 500 == 0:
                        print('Completed {} of {} utterances'.format(i_utt + 1, len(wsjmix_df)))
                with open(os.path.join(json_path,"data.json"), 'w') as f:
                    json.dump(json_data, f, indent=4)
                ##Dump
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for writing wsj0-2mix 8 k Hz and 16 kHz datasets.')
    parser.add_argument('--json-dir', type=str,
                        help='Output directory for writing wsj0-2mix 8 k Hz and 16 kHz datasets.')
    parser.add_argument('--wsj0-root', type=str,
                        help='Path to the folder containing wsj0/')
    parser.add_argument('--wham-noise-root', type=str,
                        help='Path to the downloaded and unzipped wham folder containing metadata/')
    args = parser.parse_args()
    create_wham(args.wsj0_root, args.wham_noise_root, args.output_dir, args.json_dir)
