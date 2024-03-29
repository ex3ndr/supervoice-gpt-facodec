import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import math

#
# Cached Hann Window
#

hann_window_cache = {}
def hann_window(size, device):
    global hann_window_cache
    key = str(device) + "_" + str(size)
    if key in hann_window_cache:
        return hann_window_cache[key]
    else:
        res = torch.hann_window(size).to(device)
        hann_window_cache[key] = res
        return res

#
# Mel Log Bank
#

melscale_fbank_cache = {}
def melscale_fbanks(n_mels, n_fft, f_min, f_max, sample_rate, mel_norm, mel_scale, device):
    global melscale_fbank_cache
    key = str(n_mels) + "_" + str(n_fft) + "_" + str(f_min) + "_" + str(f_max) + "_" + str(sample_rate) + "_" + str(mel_norm) + "_" + str(mel_scale) + "_"+ str(device)
    if key in melscale_fbank_cache:
        return melscale_fbank_cache[key]
    else:
        res = F.melscale_fbanks(
            n_freqs=int(n_fft // 2 + 1),
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm=mel_norm,
            mel_scale=mel_scale
        ).transpose(-1, -2).to(device)
        melscale_fbank_cache[key] = res
        return res

#
# Resampler
#

resampler_cache = {}
def resampler(from_sample_rate, to_sample_rate, device=None):
    global resampler_cache
    if device is None:
        device = "cpu"
    key = str(from_sample_rate) + "_" + str(to_sample_rate) + "_" + str(device)
    if key in resampler_cache:
        return resampler_cache[key]
    else:
        res = T.Resample(from_sample_rate, to_sample_rate).to(device)
        resampler_cache[key] = res
        return res

#
# Spectogram caclulcation
#

def spectogram(audio, n_fft, n_mels, n_hop, n_window, mel_norm, mel_scale, sample_rate):

    # Hann Window
    window = hann_window(n_window, audio.device)

    # STFT
    stft = torch.stft(audio, 
        
        # STFT Parameters
        n_fft = n_fft, 
        hop_length = n_hop, 
        win_length = n_window,
        window = window, 
        center = True,
        
        onesided = True, # Default to true to real input, but we enforce it just in case
        return_complex = True
    )

    # Compute magnitudes (|a + ib| = sqrt(a^2 + b^2)) instead of power spectrum (|a + ib|^2 = a^2 + b^2)
    # because magnitude and phase is linear to the input, while power spectrum is quadratic to the input
    # and the magnitude is easier to learn for vocoder
    # magnitudes = stft[..., :-1].abs() ** 2 # Power
    magnitudes = stft[..., :-1].abs() # Amplitude

    # Mel Log Bank
    mel_filters = melscale_fbanks(n_mels, n_fft, 0, sample_rate / 2, sample_rate, mel_norm, mel_scale, audio.device)
    mel_spec = (mel_filters @ magnitudes)

    # Log
    log_spec = torch.clamp(mel_spec, min=1e-10).log()

    return log_spec

#
# Load Mono Audio
#

def load_mono_audio(src, sample_rate, device=None):

    # Load audio
    audio, sr = torchaudio.load(src)

    # Move to device
    if device is not None:
        audio = audio.to(device)

    # Resample
    if sr != sample_rate:
        audio = resampler(sr, sample_rate, device)(audio)
        sr = sample_rate

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Convert to single dimension
    audio = audio[0]

    return audio

#
# VAD
#

vad = None

def init_if_needed():
    global vad
    if vad is None:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        vad = (model, utils)
    else:
        model, utils = vad
    return vad

def trim_silence(audio, sample_rate, padding = 0.25):

    # Load VAD
    model, utils = init_if_needed()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Get speech timestamps
    padding_frames = math.floor(sample_rate * padding)
    speech_timestamps = get_speech_timestamps(audio.unsqueeze(0), model.to(audio.device), sampling_rate=sample_rate)    
    if len(speech_timestamps) > 0:
        voice_start = speech_timestamps[0]['start'] - padding_frames
        voice_end = speech_timestamps[-1]['end'] + padding_frames
        voice_start = max(0, voice_start)
        voice_end = min(len(audio), voice_end)
        audio = audio[voice_start:voice_end]

    return audio