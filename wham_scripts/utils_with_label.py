import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_scaled_wav(path, scaling_factor, downsample_8K=False):
    samples, sr_orig = sf.read(path)

    if len(samples.shape) > 1:
        samples = samples[:, 0]

    if downsample_8K:
        samples = resample_poly(samples, 8000, sr_orig)
    samples *= scaling_factor
    return samples


def wavwrite_quantize(samples):
    return np.int16(np.round((2 ** 15) * samples))


def quantize(samples):
    int_samples = wavwrite_quantize(samples)
    return np.float64(int_samples) / (2 ** 15)


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype='PCM_16')


def append_or_truncate(s1_samples, s2_samples, noise_samples, mixture_timestamps, min_or_max='max', start_samp_16k=0, downsample=False):
    if downsample:
        speech_start_sample = start_samp_16k // 2
    else:
        speech_start_sample = start_samp_16k
    #print("AT: ",mixture_timestamps)
    #print(len(noise_samples))
    speech_end_sample = speech_start_sample + len(s1_samples)
    #print(speech_start_sample," ",speech_end_sample)
    if min_or_max == 'min':
        noise_samples = noise_samples[speech_start_sample:speech_end_sample]
        mixture_label_samples = np.zeros((2,len(noise_samples)))
        if downsample:
            start_0 = int(8000*mixture_timestamps[0][1])
            end_0 = int(8000*mixture_timestamps[0][2])
            start_1 = int(8000*mixture_timestamps[1][1])
            end_1 = int(8000*mixture_timestamps[1][2])
        else:
            start_0 = int(16000*mixture_timestamps[0][1])
            end_0 = int(16000*mixture_timestamps[0][2])
            start_1 = int(16000*mixture_timestamps[1][1])
            end_1 = int(16000*mixture_timestamps[1][2])
    else:
        s1_append = np.zeros_like(noise_samples)
        s2_append = np.zeros_like(noise_samples)
        s1_append[speech_start_sample:speech_end_sample] = s1_samples
        s2_append[speech_start_sample:speech_end_sample] = s2_samples
        s1_samples = s1_append
        s2_samples = s2_append
        ######
        mixture_label_samples = np.zeros((2,len(noise_samples)))
        if downsample:
            start_0 = int(speech_start_sample+8000*mixture_timestamps[0][1])
            end_0 = int(speech_start_sample+8000*mixture_timestamps[0][2])
            start_1 = int(speech_start_sample+8000*mixture_timestamps[1][1])
            end_1 = int(speech_start_sample+8000*mixture_timestamps[1][2])
        else:
            start_0 = int(speech_start_sample+16000*mixture_timestamps[0][1])
            end_0 = int(speech_start_sample+16000*mixture_timestamps[0][2])
            start_1 = int(speech_start_sample+16000*mixture_timestamps[1][1])
            end_1 = int(speech_start_sample+16000*mixture_timestamps[1][2])
        mixture_label_samples[0][start_0:end_0] = 1
        mixture_label_samples[1][start_1:end_1] = 1
        #######
    #print("AL : ",mixture_label_samples)
    return s1_samples, s2_samples, noise_samples, mixture_label_samples


def fix_length(s1, s2, s1_label, s2_label, min_or_max='max'):
    # Fix length
    mixture_label_time = np.zeros((2,4))
    if(len(s2_label)>3):
        mixture_label_time[1][1] = s2_label[0]
        mixture_label_time[1][-1] = s2_label[-1]
        mixture_label_time[1][-2] = s2_label[-2]
    elif(len(s1_label)>3):
        mixture_label_time[0][1] = s1_label[0]
        mixture_label_time[0][-1] = s1_label[-1]
        mixture_label_time[0][-2] = s1_label[-2]
    else:
        mixture_label_time[0][1:] = s1_label
        mixture_label_time[1][1:] = s2_label
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
        ######
        if(s1_label[-1]<s2_label[-1]): ##First Speaker Ends First
            mixture_label_time[1][-1] = mixture_label_time[0][-1]
            for i in range(len(mixture_label_time[1])):
                if mixture_label_time[1][i] < mixture_label_time[0][-1]:
                    mixture_label_time[1][i] = mixture_label_time[0][-1]
        else: 
            mixture_label_time[0][-1] = mixture_label_time[1][-1]
            for i in range(len(mixture_label_time[0])):
                if mixture_label_time[0][i] < mixture_label_time[1][-1]:
                    mixture_label_time[0][i] = mixture_label_time[1][-1]           
        ######
    else:  # max
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
        if(s1_label[-1]<s2_label[-1]): ##First Speaker Ends First
            mixture_label_time[0][-1] = mixture_label_time[1][-1]
        else:
            mixture_label_time[1][-1] = mixture_label_time[0][-1]
    #print("FL: ",s1_label)
    #print("FL: ",s2_label)
    #print("FL: ",mixture_label_time )
    return s1, s2, mixture_label_time 


def create_wham_mixes(s1_samples, s2_samples, noise_samples):
    mix_clean = s1_samples + s2_samples
    mix_single = noise_samples + s1_samples
    mix_both = noise_samples + s1_samples + s2_samples
    return mix_clean, mix_single, mix_both