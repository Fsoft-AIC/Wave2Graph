import numpy as np
import librosa
import os
from tqdm import tqdm

def create_mel_raw(current_window, sample_rate, n_mels=128, nfft=2048, hop=512, resz=1):
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S-S.min()) / (S.max() - S.min())
    S *= 255
    return S

def func(y, sample_rate=16_000, n_mfcc=39, n_fft=1024, hop_length=200):
    y, _ = librosa.effects.trim(y, frame_length=n_fft, hop_length=hop_length)

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, dct_type=2)
    pearson = np.nan_to_num(np.corrcoef(mfcc))
    return mfcc, pearson

def crossSpectrum(x, y, nperseg=1):
    #-------------------Remove mean-------------------
    cross = np.zeros(nperseg, dtype='complex128')
    if nperseg < 2:
        cfx = np.fft.fft(x - np.mean(x))
        cfy = np.fft.fft(y - np.mean(y))

        # Get cross spectrum
        cross = np.sum(cfx.conj()*cfy)
    else:
        for ind in range(x.size // nperseg):
            xp = x[ind * nperseg: (ind + 1)*nperseg]
            yp = y[ind * nperseg: (ind + 1)*nperseg]

            xp = xp - np.mean(xp)
            yp = yp - np.mean(yp)

            # Do FFT
            cfx = np.fft.fft(xp)
            cfy = np.fft.fft(yp)

            # Get cross spectrum
            cross += cfx.conj()*cfy
    freq = np.fft.fftfreq(nperseg)
    return cross, freq


def create_coherence(feature, n_mfcc):
    coh = np.zeros([n_mfcc, n_mfcc])
    for m1 in range(n_mfcc):
        for m2 in range(n_mfcc):
            p11, freq = crossSpectrum(feature[m1], feature[m1])
            p22, freq = crossSpectrum(feature[m2], feature[m2])
            p12, freq = crossSpectrum(feature[m1], feature[m2])
            f = np.abs(p12)**2/p11.real/p22.real
            # f, Cxy = signal.coherence(feature[m1], feature[m2], sr, nperseg=1)
            # plt.semilogy(f, Cxy)
            # plt.xlabel('frequency [Hz]')
            # plt.ylabel('Coherence')
            # plt.show()
            coh[m1, m2] = f
    return coh

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)

args = parser.parse_args()

n_mfcc = 39
if '13' in args.dir:
    n_mfcc = 13
elif '39' in args.dir:
    n_mfcc = 39
elif '64' in args.dir:
    n_mfcc = 64
elif '128' in args.dir:
    n_mfcc = 128

for dir in ['cn', 'ad']:
    for path in tqdm(os.listdir(dir)):
        if not path.endswith('.wav'):
            continue
        path = dir + '/' + path
        data, _ = librosa.load(path, sr = 16_000)
        max_len = 160_000
        N = data.shape[0]
        K = 5
        gap = (N-max_len)//K

        curr_mfcc = []
        curr_corr = []
        curr_mel = []
        
        for i in range(K):
            if 'mel' in args.dir:
                mfcc = create_mel_raw(data[gap*i:gap*i+max_len], 16_000, n_mels=39)
                pearson = np.nan_to_num(np.corrcoef(mfcc))
            else:
                mfcc, pearson = func(data[gap*i:gap*i+max_len], n_mfcc=n_mfcc)
            if 'cohe' in args.dir:
                pearson = create_coherence(mfcc, n_mfcc)
            # np.save(path[:-4] + f'_mfcc_{i}.npy', mfcc) # 39xT
            # np.save(path[:-4] + f'_corr_{i}.npy', pearson) #39x39

            mel = create_mel_raw(data[gap*i:gap*i+max_len], 16_000) # 128xT
            curr_mfcc.append(mfcc)
            curr_corr.append(pearson)
            curr_mel.append(mel)

        curr_mfcc = np.concatenate(curr_mfcc, axis = 1) # 39x(Tx5)
        curr_corr = np.concatenate(curr_corr, axis = 1) # 39x(39x5)
        curr_mel = np.concatenate(curr_mel, axis = 1) # 128x(Tx5)

        path = os.path.join(args.dir, path)

        np.save(path[:-4] + '_mfcc.npy', curr_mfcc)
        np.save(path[:-4] + '_corr.npy', curr_corr)
        np.save(path[:-4] + '_mel.npy', curr_mel)