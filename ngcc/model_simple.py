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
                                        normalize_input=True, normalize_output=False, pool_len=5, use_mel=True, n_mics=4):
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

        self.n_mics = n_mics
        self.max_tau = max_tau
        self.n_delays = 2 * max_tau + 1
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.pool_len = pool_len
        self.n_mel_bins = n_mel_bins
        self.use_mel = use_mel
        self.num_combinations = int(self.n_mics * (self.n_mics - 1) / 2)

        sincnet_params = {'input_dim': sig_len,
                          'fs': fs,
                          'cnn_N_filt': [num_channels],
                          'cnn_len_filt': [sig_len-1],
                          'cnn_max_pool_len': [1],
                          'cnn_use_laynorm_inp': False,
                          'cnn_use_batchnorm_inp': False,
                          'cnn_use_laynorm': [False],
                          'cnn_use_batchnorm': [True],
                          'cnn_act': ['linear'],
                          'cnn_drop': [0.0],
                          'use_sinc': use_sinc,
                          }

        self.backbone = SincNet(sincnet_params)
        self.pool = torch.nn.AvgPool2d((pool_len, 1)) # torch.nn.AvgPool1d(pool_len)
        #self.mlp_kernels = [11, 9, 7]
        #self.channels = [num_channels, num_channels, num_channels, num_channels]
        self.final_kernel = 3

        self.gcc = GCC(max_tau=self.max_tau, dim=4, filt='phat')

        #self.mlp = nn.ModuleList([nn.Sequential(
        #        nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size=k),
        #        nn.BatchNorm1d(self.channels[i+1]),
        #        nn.LeakyReLU(0.2)) for i, k in enumerate(self.mlp_kernels)])
        

        self.cc_feat1 = nn.Sequential(
                nn.Linear(self.n_delays, 4*self.n_delays),
                nn.LayerNorm(4*self.n_delays),
                nn.GELU(),
                nn.Dropout(0.5)
        )

        self.cc_feat2 = nn.Sequential(
                nn.Linear(num_channels*4*self.n_delays, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.5)
        )

        self.cc_feat3 = nn.Sequential(
                nn.Linear(self.num_combinations * 32, 128),
                nn.LayerNorm(128),
                nn.GELU()
        )

        self.spec_feature = nn.Sequential(
                nn.Conv1d(num_channels, 64, kernel_size=5, stride=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=5, stride=2),
                nn.BatchNorm1d(128),
                nn.GELU()
        )

        self.spec_feature2 = nn.Sequential(
                nn.Conv1d(128*self.n_mics, 128, kernel_size=5, stride=2),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=5, stride=2),
                nn.BatchNorm1d(128),
                nn.GELU()
        )


        self.combined_feature = nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128)
        )
        
        
        #self.final_conv = nn.Conv1d(num_channels, num_out_channels, kernel_size=self.final_kernel)
        #self.spec_conv = nn.Conv1d(num_channels, num_out_channels, kernel_size=self.final_kernel, stride=self.final_kernel)
        self.cc_drop = nn.Dropout(0.5)

        #self.spec_conv = nn.Sequential(
                #nn.Conv1d(num_channels, num_out_channels, self.final_kernel),
                #nn.BatchNorm1d(num_out_channels),
                #nn.LeakyReLU(0.2),
                #n.Dropout(0.5)
        #)

        
        #if self.use_mel:
        #    self.nfft = next_greater_power_of_2(sig_len)
        #    self.mel_transform = torchaudio.transforms.MelScale(n_mels=self.n_mel_bins, sample_rate=fs, n_stft=self.nfft//2+1)

        #else:
        #    in_size = sig_len // self.final_kernel
        #    self.proj = nn.Sequential(
        #            nn.Dropout(0.5),
        #            nn.Linear(in_size, in_size // 2),
        #            nn.LayerNorm(in_size // 2),
        #            nn.GELU(),
        #            nn.Dropout(0.5),
        #            nn.Linear(in_size // 2, self.n_mel_bins)
        #    )

    def forward(self, audio):

        if self.normalize_input:
            audio /= audio.std(dim=-1, keepdims=True)


        # filter signals 
        B, M, T, L = audio.shape # (batch_size, #mics, #time_windows, win_len)
        x = audio.reshape(-1, 1, T*L)
        x = self.backbone(x)
        #x = self.pool(x)

        #s = x.shape[2]
        #padding = get_pad(
        #    size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
        #x_spec = F.pad(x, pad=padding, mode='constant')
        #x_spec = self.spec_conv(x)
        _, C, _ = x.shape
        x = x.reshape(B, M, C, T*L).reshape(B, M, C, T, L).permute(0, 1, 3, 2, 4) #(batch_size, #mics, #time_windows, channels, win_len)
        x_spec = x.reshape(-1, C, L)
        x_spec = self.spec_feature(x_spec)
        _, C_spec, _ = x_spec.shape
        x_spec = x_spec.reshape(B, M, T, C_spec, -1).permute(0, 2, 1, 3, 4)
        x_spec = x_spec.reshape(B*T, M*C_spec, -1)
        x_spec = self.spec_feature2(x_spec)
        x_spec, _ = torch.max(x_spec, dim=-1)
        x_spec = x_spec.reshape(B, T, -1)

        

        #_, C, _ = x.shape
        #T = int(T / self.pool_len)
        #L_spec = int(L // self.final_kernel)
        #x_cc = x.reshape(B, M, C, T*L) # (batch_size, #mics, channels, #time_windows * win_len)
        #x_cc = x.reshape(B, M, C, T, L).permute(0, 1, 3, 2, 4) # (batch_size, #mics, #time_windows, channels, win_len)

        #_, C_spec = x_spec.shape
        #x_spec = x_spec.reshape(B, M, C_spec, -1) # (batch_size, #mics, channels, #time_windows * win_len)
        #x_spec = x_spec..permute(0, 1, 3, 2, 4) # (batch_size, #mics, #time_windows, channels, win_len)

        cc = [] 
        # compute gcc-phat for pairwise microphone combinations
        for m1 in range(0, M):
            for m2 in range(m1+1, M):
                
                y1 = x[:, m1, :, :, :]
                y2 = x[:, m2, :, :, :]
                cc1 = self.gcc(y1, y2) # (batch_size, #time_windows, channels, #delays)
                #cc2 = torch.flip(cc1, dims=[-1]) # if we have cc(m1, m2), do we need cc(m2,m1)?
                cc.append(cc1)
                #cc.append(cc2)

        cc = torch.stack(cc, dim=-1) # (batch_size, #time_windows, channels, #delays, #combinations)
        
        cc = cc.permute(0, 1, 4, 2, 3) # (batch_size, #time_windows, #combinations, channels, #delays)
        cc = self.cc_feat1(cc)
        cc = cc.reshape(B, T, self.num_combinations, -1)
        cc = self.cc_feat2(cc)
        cc = cc.reshape(B, T, -1)
        cc = self.cc_feat3(cc)

        #cc = cc[:, :, :, :, 1:] #throw away one dely to make #delays even

        #B, N, _, C, tau = cc.shape
        #cc = cc.reshape(-1, C, tau)
        
        #cc = self.cc_feat(cc)
        #_, C, cc_dim = cc.shape
        #cc = cc.reshape(-1, C*cc_dim)
        #cc = self.cc_feat2(cc)

        
        #for k, layer in enumerate(self.mlp):
        #    s = cc.shape[2]
        #    padding = get_pad(
        #        size=s, kernel_size=self.mlp_kernels[k], stride=1, dilation=1)
        #    cc = F.pad(cc, pad=padding, mode='constant')
        #    cc = layer(cc)

        #s = cc.shape[2]
        #padding = get_pad(
        #    size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
        #cc = F.pad(cc, pad=padding, mode='constant')
        #cc = self.final_conv(cc)

        #_, C, tau = cc.shape
        #cc = cc.reshape(B, N, T, C, tau)
        #cc = cc.permute(0, 1, 3, 2, 4)  # (batch_size, #combinations, channels, #time_windows, #delays)
        #cc = cc.reshape(B, N * C, T, tau) # (batch_size, #combinations * channels, #time_windows, #delays)
        #cc = self.cc_drop(cc)

        #if self.normalize_output:
        #    cc /= cc.std(dim=-1, keepdims=True)

        # compute log mel-spectrograms from x
        #x_spec = x_spec.permute(0, 1, 3, 2, 4) # (batch_size, #mics, channels, #time_windows, #delays)
        #x_spec = x_spec.reshape(B, M * C_spec, T, L_spec) # (batch_size, #mics * channels, #time_windows, #delays)
        
        #if self.use_mel:
        #    X = torch.fft.rfft(x_spec, n=self.nfft, norm='ortho', dim=-1) # (batch_size, ##mics * channels, #time_windows, #freqs)

        #    mag_spectra = torch.abs(X)**2 # 
        #    mel_spectra = self.mel_transform(mag_spectra.permute(0,1,3,2)).permute(0,1,3,2)
        #else:
        #    mel_spectra = self.proj(x_spec)

        # here, #mel_weights must be equal to #delays for this to work
        feat = torch.cat((x_spec, cc), dim=-1) # (batch_size, ##mics * channels + #combinations, #time_windows, #mel_weights)
        feat = self.combined_feature(feat)
        # pool over time
        feat = self.pool(feat)

        return feat



