import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_same_pad import get_pad
from ngcc.dnn_models import SincNet
import torch.fft
import librosa
import torchaudio

def next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

class GCC(nn.Module):
    def __init__(self, max_tau=None, dim=2, filt='phat', epsilon=0.001, beta=None):
        super().__init__()

        ''' GCC implementation based on Knapp and Carter,
        "The Generalized Correlation Method for Estimation of Time Delay",
        IEEE Trans. Acoust., Speech, Signal Processing, August, 1976 '''

        self.max_tau = max_tau
        self.dim = dim
        self.filt = filt
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, x, y):

        n = x.shape[-1] + y.shape[-1]

        # Generalized Cross Correlation Phase Transform
        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)

        if self.filt == 'phat':
            phi = 1 / (torch.abs(Gxy) + self.epsilon)

        elif self.filt == 'roth':
            phi = 1 / (X * torch.conj(X) + self.epsilon)

        elif self.filt == 'scot':
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            phi = 1 / (torch.sqrt(X * Y) + self.epsilon)

        elif self.filt == 'ht':
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            gamma = Gxy / torch.sqrt(Gxx * Gxy)
            phi = torch.abs(gamma)**2 / (torch.abs(Gxy)
                                         * (1 - gamma)**2 + self.epsilon)

        elif self.filt == 'cc':
            phi = 1.0

        else:
            raise ValueError('Unsupported filter function')

        if self.beta is not None:
            cc = []
            for i in range(self.beta.shape[0]):
                cc.append(torch.fft.irfft(
                    Gxy * torch.pow(phi, self.beta[i]), n))

            cc = torch.cat(cc, dim=1)

        else:
            cc = torch.fft.irfft(Gxy * phi, n)

        max_shift = int(n / 2)
        if self.max_tau:
            max_shift = np.minimum(self.max_tau, int(max_shift))

        if self.dim == 2:
            cc = torch.cat((cc[:, -max_shift:], cc[:, :max_shift+1]), dim=-1)
        elif self.dim == 3:
            cc = torch.cat(
                (cc[:, :, -max_shift:], cc[:, :, :max_shift+1]), dim=-1)
        elif self.dim == 4:
            cc = torch.cat(
                (cc[:, :, :, -max_shift:], cc[:, :, :, :max_shift+1]), dim=-1)

        return cc


class NGCCPHAT(nn.Module):
    def __init__(self, max_tau=64, n_mel_bins=64, use_sinc=True,
                                        sig_len=960, num_channels=128, num_out_channels=8, fs=24000,
                                        normalize_input=True, normalize_output=False, pool_len=5):
        super().__init__()

        '''
        Neural GCC-PHAT with SincNet backbone

        arguments:
        max_tau - the maximum possible delay considered
        use_sinc - use sincnet backbone if True, otherwise use regular conv layers
        sig_len - length of input signal
        n_channel - number of gcc correlation channels to use
        fs - sampling frequency
        '''

        print(sig_len)
        print(fs)
        print(max_tau)
        self.max_tau = max_tau
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.pool_len = pool_len

        sincnet_params = {'input_dim': sig_len,
                          'fs': fs,
                          'cnn_N_filt': [num_channels, num_channels, num_channels, num_channels],
                          'cnn_len_filt': [sig_len-1, 11, 9, 7],
                          'cnn_max_pool_len': [pool_len, 1, 1, 1],
                          'cnn_use_laynorm_inp': False,
                          'cnn_use_batchnorm_inp': False,
                          'cnn_use_laynorm': [False, False, False, False],
                          'cnn_use_batchnorm': [True, True, True, True],
                          'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'linear'],
                          'cnn_drop': [0.0, 0.0, 0.0, 0.0],
                          'use_sinc': use_sinc,
                          }

        self.backbone = SincNet(sincnet_params)
        self.mlp_kernels = [11, 9, 7]
        self.channels = [num_channels, num_channels, num_channels, num_channels]
        self.final_kernel = [5]

        self.gcc = GCC(max_tau=self.max_tau, dim=4, filt='phat')

        self.mlp = nn.ModuleList([nn.Sequential(
                nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size=k),
                nn.BatchNorm1d(self.channels[i+1]),
                nn.LeakyReLU(0.2)) for i, k in enumerate(self.mlp_kernels)])
        

        self.final_conv = nn.Conv1d(num_channels, num_out_channels, kernel_size=self.final_kernel)

        self.spec_conv = nn.Sequential(
                nn.Conv1d(num_channels, num_out_channels, self.final_kernel),
                nn.BatchNorm1d(num_out_channels),
                nn.LeakyReLU(0.2),
                #n.Dropout(0.5)
        )

        self.n_mel_bins = n_mel_bins
        self.nfft = next_greater_power_of_2(sig_len)
        self.mel_transform = torchaudio.transforms.MelScale(n_mels=self.n_mel_bins, sample_rate=fs, n_stft=self.nfft//2+1)


    def forward(self, audio):

        if self.normalize_input:
            audio /= audio.std(dim=-1, keepdims=True)


        # filter signals 
        B, M, T, L = audio.shape # (batch_size, #mics, #time_windows, win_len)
        x = audio.reshape(-1, 1, T*L)
        print("input")
        print(torch.sum(torch.isnan(x.flatten())))
        x = self.backbone(x)
        print("backbone")
        print(torch.sum(torch.isnan(x.flatten())))

        s = x.shape[2]
        padding = get_pad(
            size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
        x_spec = F.pad(x, pad=padding, mode='constant')
        x_spec = self.spec_conv(x_spec)

        _, C, _ = x.shape
        T = int(T / self.pool_len)
        x_cc = x.reshape(B, M, C, T*L) # (batch_size, #mics, channels, #time_windows * win_len)
        x_cc = x.reshape(B, M, C, T, L).permute(0, 1, 3, 2, 4) # (batch_size, #mics, #time_windows, channels, win_len)

        _, C_spec, _ = x_spec.shape
        x_spec = x_spec.reshape(B, M, C_spec, T*L) # (batch_size, #mics, channels, #time_windows * win_len)
        x_spec = x_spec.reshape(B, M, C_spec, T, L).permute(0, 1, 3, 2, 4) # (batch_size, #mics, #time_windows, channels, win_len)

        cc = [] 
        # compute gcc-phat for pairwise microphone combinations
        for m1 in range(0, M):
            for m2 in range(m1+1, M):
                
                y1 = x_cc[:, m1, :, :, :]
                y2 = x_cc[:, m2, :, :, :]
                cc1 = self.gcc(y1, y2) # (batch_size, #time_windows, channels, #delays)
                #cc2 = torch.flip(cc1, dims=[-1]) # if we have cc(m1, m2), do we need cc(m2,m1)?
                cc.append(cc1)
                #cc.append(cc2)

        cc = torch.stack(cc, dim=-1) # (batch_size, #time_windows, channels, #delays, #combinations)
        cc = cc.permute(0, 4, 1, 2, 3) # (batch_size, #combinations, #time_windows, channels, #delays)
        cc = cc[:, :, :, :, 1:] #throw away one dely to make #delays even
        
        print("CC")
        print(torch.sum(torch.isnan(cc.flatten())))

        B, N, _, C, tau = cc.shape
        cc = cc.reshape(-1, C, tau)
        for k, layer in enumerate(self.mlp):
            s = cc.shape[2]
            padding = get_pad(
                size=s, kernel_size=self.mlp_kernels[k], stride=1, dilation=1)
            cc = F.pad(cc, pad=padding, mode='constant')
            cc = layer(cc)

        s = cc.shape[2]
        padding = get_pad(
            size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
        cc = F.pad(cc, pad=padding, mode='constant')
        cc = self.final_conv(cc)

        _, C, tau = cc.shape
        cc = cc.reshape(B, N, T, C, tau)
        cc = cc.permute(0, 1, 3, 2, 4)  # (batch_size, #combinations, channels, #time_windows, #delays)
        cc = cc.reshape(B, N * C, T, tau) # (batch_size, #combinations * channels, #time_windows, #delays)

        if self.normalize_output:
            cc /= cc.std(dim=-1, keepdims=True)

        # compute log mel-spectrograms from x
        #print(x_spec.shape)
        x_spec = x_spec.permute(0, 1, 3, 2, 4) # (batch_size, #mics, channels, #time_windows, #delays)
        #print(x_spec.shape)
        x_spec = x_spec.reshape(B, M * C_spec, T, L) # (batch_size, #mics * channels, #time_windows, #delays)
        print("x_spec")
        print(torch.sum(torch.isnan(x_spec.flatten())))
        X = torch.fft.rfft(x_spec, n=self.nfft, norm='ortho', dim=-1) # (batch_size, ##mics * channels, #time_windows, #freqs)
        print("X")
        print(torch.sum(torch.isnan(X.flatten())))

        mag_spectra = torch.abs(X)**2 # 
        mel_spectra = self.mel_transform(mag_spectra.permute(0,1,3,2)).permute(0,1,3,2)
        #mel_spectra =  (mag_spectra*self.mel_wts).sum(axis = -1) # (batch_size, ##mics * channels, #time_windows, #mel_weights)
        #log_mel_spectra = librosa.power_to_db(mel_spectra)
        print("mel_spectra")
        print(torch.sum(torch.isnan(mel_spectra.flatten())))

        # here, #mel_weights must be equal to #delays for this to work
        feat = torch.cat((mel_spectra, cc), dim=1) # (batch_size, ##mics * channels + #combinations, #time_windows, #mel_weights)

        return feat



