#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import parameters
import time
from time import gmtime, strftime
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model 
from model import NGCCModel
from speechbrain.nnet.losses import PitWrapper 
from torch_audiomentations import AddColoredNoise
from cst_former.CST_former_model import CST_former
from torchsummary import summary
from warmup_scheduler import GradualWarmupScheduler

def deg2rad(deg):
    return deg * 2 * np.pi / 360


def rad2deg(rad):
    return rad * 360 / (2 * np.pi)

def center_mic_coords(mic_coords, mic_center):
    mic_locs = np.empty((0, 3))
    for coord in mic_coords:
        rad, azi, ele = coord
        azi = deg2rad(azi)
        ele = deg2rad(ele)
        x_offset = rad * np.cos(azi) * np.cos(ele)
        y_offset = rad * np.sin(azi) * np.cos(ele)
        z_offset = rad * np.sin(ele)
        mic_loc = mic_center + np.array([x_offset, y_offset, z_offset])
        mic_locs = np.vstack([mic_locs, mic_loc])
    return mic_locs


class TdoaLoss(nn.Module):
    def __init__(self, fs=24000, c=343, nmics=4, ntdoas=6, max_tau=6, tracks=5):#, max_events=3):
        super(TdoaLoss, self).__init__()
        loss_module = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.pit_loss = PitWrapper(loss_module)
        self.fs = fs
        self.c = c
        self.nmics = nmics
        self.ntdoas = ntdoas
        self.max_tau = max_tau
        self.tracks = tracks 
        #self.max_events = max_events

        m1_coords = [0.042, 45, 35]
        m2_coords = [0.042, -45, -35]
        m3_coords = [0.042, 135, -35]
        m4_coords = [0.042, -135, 35]

        mic_coords = [m1_coords, m2_coords, m3_coords, m4_coords]
        mic_center = [0.0, 0.0, 0.0]

        self.mic_locs = torch.Tensor(center_mic_coords(mic_coords, mic_center))

    def get_tdoa_target(self, target):
        B, T, _, F, C = target.shape
        Tr = 5 # up to 5 events is possible
        tdoas = torch.zeros((B, T, Tr, self.ntdoas))
        tdoas[:] = torch.nan
        ignore_idx = int(-100)
        max_tau = int(self.max_tau)
        for b in range(B):
            for t in range(T):
                tr_cnt = 0
                n_active = 0
                for tr in range(Tr):
                    for c in range(C):
                        if tr_cnt >= Tr:
                            break
                        active = target[b, t, tr, 0, c]
                        if active:
                            n_active +=1
                            doa = target[b, t, tr, 1:4, c].squeeze()
                            dist = target[b, t, tr, 4, c]
                            source_loc = doa * dist
                            cnt = 0
                            for m1 in range(self.nmics):
                                for m2 in range(m1+1, self.nmics):
                                    mic1 = self.mic_locs[m1]
                                    mic2 = self.mic_locs[m2]
                                    tdoa = torch.sqrt(torch.sum((source_loc-mic1)**2)) - torch.sqrt(torch.sum((source_loc-mic2)**2))
                                    tdoa = int(torch.round(tdoa*self.fs/self.c))
                                    tdoas[b, t, tr_cnt, cnt] = tdoa+max_tau
                                    cnt +=1
                            tr_cnt +=1
                #if n_active > 1:
                #    print(n_active, flush=True)
                if n_active == 0:
                    tdoas[b, t, :, :] = ignore_idx
                elif tr_cnt < Tr and n_active > 0:
                    tdoas[b, t, tr_cnt:, :] = ignore_idx# tdoas[b, t, tr_cnt-1, :]

        #randomly shuffle the events
        tdoas = np.swapaxes(tdoas, 0, 2)
        np.random.shuffle(tdoas) # shuffle along axis 0 (track axis)
        tdoas = np.swapaxes(tdoas, 0, 2)

        return tdoas[:, :, :self.tracks] # return only self.tracks tdoas per time slot
    
    def forward(self, pred, target):
        self.mic_locs = self.mic_locs.to(target.device)
        #print("calculating target")
        tdoa_target = self.get_tdoa_target(target)
        B, T, C, Tr, ntdoa = pred.shape
        tdoa_target = tdoa_target.permute(0, 1, 3, 2).reshape(B*T*ntdoa, Tr).long().to(pred.device)  
        pred = pred.permute(0, 1, 4, 2, 3).reshape(B*T*ntdoa, C, Tr)   
        
        loss, opt_p = self.pit_loss(pred.unsqueeze(1), tdoa_target.unsqueeze(1))
        acc = 0.
        n_pred = 0.
        for b in range(tdoa_target.shape[0]):
            this_pred = pred[b].argmax(dim=0, keepdim=False)
            this_target = tdoa_target[b, list(opt_p[b])]
            valid_idx = this_target!=-100
            this_target = this_target[valid_idx]
            this_pred = this_pred[valid_idx]
            if not this_pred.nelement() == 0:
                acc += this_pred.eq(this_target.view_as(this_pred)).float().sum().item()
                n_pred += this_pred.nelement()
        
        if n_pred > 0:
            acc = acc / n_pred
        else:
            acc = 0.

        return loss.mean(), acc

def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5
      
    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    dist0 = accdoa_in[:, :, 3*nb_classes:4*nb_classes]
    dist0[dist0 < 0.] = 0.
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes], accdoa_in[:, :, 6*nb_classes:7*nb_classes]
    dist1 = accdoa_in[:, :, 7*nb_classes:8*nb_classes]
    dist1[dist1<0.] = 0.
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 4*nb_classes: 7*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 8*nb_classes:9*nb_classes], accdoa_in[:, :, 9*nb_classes:10*nb_classes], accdoa_in[:, :, 10*nb_classes:11*nb_classes]
    dist2 = accdoa_in[:, :, 11*nb_classes:]
    dist2[dist2<0.] = 0.
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 8*nb_classes:11*nb_classes]

    return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device, criterion_tdoa=None):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    test_filelist = data_generator.get_filelist()

    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for values in data_generator.generate():
            if len(values) == 2:
                data, target = values
                #print(data.shape)
                #print(target.shape)
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
                bs = params['batch_size']
                if data.shape[0] > bs and params['raw_chunks']:
                    max_cnt = data.shape[0] // bs
                    output = []
                    output_tdoa = []
                    for cnt in range(0, max_cnt):
                        this_data = data[cnt*bs:(cnt+1)*bs]
                        #print(this_data.shape)
                        if criterion_tdoa is not None:
                            this_output, this_output_tdoa = model(this_data)
                            output.append(this_output)
                            output_tdoa.append(this_output_tdoa)
                        else:
                            this_output = model(this_data)
                            #print(this_output.shape)
                            output.append(this_output)
                    
                    this_data = data[(cnt+1)*bs:]
                    #print(this_data.shape)
                    if criterion_tdoa is not None:
                        this_output, this_output_tdoa = model(this_data)
                        output.append(this_output)
                        output_tdoa.append(this_output_tdoa)
                        output_tdoa = torch.cat(output_tdoa, dim=0)
                    else:
                        this_output = model(this_data)
                        #print(this_output.shape)
                        output.append(this_output)
                    
                    output = torch.cat(output, dim=0)
                    

                else:
                    if criterion_tdoa is not None:
                        output, output_tdoa = model(data)
                    else:
                        output = model(data)
            elif len(values) == 3:
                data, vid_feat, target = values
                data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()
                output = model(data, vid_feat)

            if criterion_tdoa is not None:
                loss1 = criterion(output, target)
                loss2, acc = criterion_tdoa(output_tdoa, target)
                loss = (1.0 - params['lambda']) * loss1 + params['lambda'] * loss2
            else:
                loss = criterion(output, target)

            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                dist_pred0 = reshape_3Dto2D(dist_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                dist_pred1 = reshape_3Dto2D(dist_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
                dist_pred2 = reshape_3Dto2D(dist_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file

            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                            if sed_pred1[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                            if sed_pred2[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]]) 
            data_generator.write_output_format_file(output_file, output_dict)


            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break


        test_loss /= nb_test_batches

    return test_loss


def train_epoch(data_generator, optimizer, model, criterion, params, device, criterion_tdoa):
    nb_train_batches, train_loss = 0, 0.
    model.train()

    train_transform = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(7, iid_masks=True),
            torchaudio.transforms.FrequencyMasking(7, iid_masks=True),
    )

    augment = AddColoredNoise(p=1.0, min_snr_in_db=5, max_snr_in_db=30, sample_rate=params['fs'], mode="per_channel", p_mode="per_channel")           

    tdoa_loss_ma = -1
    tdoa_acc_ma = -1
    for values in data_generator.generate():
        # load one batch of data
        if len(values) == 2:
            data, target = values
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
            if params['specaugment'] and not params['raw_chunks']:
                spec = data[:, :params['n_mics']].permute(0, 1, 3, 2)
                data[:, :params['n_mics']] = train_transform(spec).permute(0, 1, 3, 2)
            
            if params['augment'] and params['raw_chunks']:
                B, C, T, L = data.shape
                data = data.permute(0, 2, 1, 3).reshape(-1, C, L)
                data = augment(data)
                data = data.reshape(B, T, C, L).permute(0, 2, 1, 3)
            
            optimizer.zero_grad()
            if criterion_tdoa is not None:
                output, output_tdoa = model(data)
            else:
                output = model(data)
        elif len(values) == 3:
            data, vid_feat, target = values
            data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()
            
            if params['specaugment'] and not params['raw_chunks']:
                spec = data[:, :params['n_mics']].permute(0, 1, 3, 2)
                data[:, :params['n_mics']] = train_transform(spec).permute(0, 1, 3, 2)

            if params['augment'] and params['raw_chunks']:
                B, C, T, L = data.shape
                data = data.permute(0, 2, 1, 3).reshape(-1, C, L)
                data = augment(data)
                data = data.reshape(B, T, C, L).permute(0, 2, 1, 3)

            optimizer.zero_grad()
            output = model(data, vid_feat)
                    
        if criterion_tdoa is not None:
            loss1 = criterion(output, target)
            loss2, acc = criterion_tdoa(output_tdoa, target)
            if tdoa_loss_ma == -1:
                tdoa_loss_ma = loss2.item()
                tdoa_acc_ma = acc
            else:
                tdoa_loss_ma = 0.95 * tdoa_loss_ma + 0.05 * loss2.item()
                if acc > 0:
                    tdoa_acc_ma = 0.95 * tdoa_acc_ma + 0.05 * acc
            print("tdoa loss: " + str(tdoa_loss_ma)+ ", tdoa acc: " + str(tdoa_acc_ma), flush=True)
            loss = (1.0 - params['lambda']) * loss1 + params['lambda'] * loss2
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break


    train_loss /= nb_train_batches

    return train_loss


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    LOG_FOUT = open(os.path.join('logs/', job_id+'.txt'), 'w')

    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str, flush=True)

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        if '2020' in params['dataset_dir']:
            test_splits = [1]
            val_splits = [2]
            train_splits = [[3, 4, 5, 6]]

        elif '2021' in params['dataset_dir']:
            test_splits = [6]
            val_splits = [5]
            train_splits = [[1, 2, 3, 4]]

        elif '2022' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 
        elif '2023' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 
        elif '2024' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3, 9]]# [[1, 2, 3, 9]] # split 1 and 2 are simulated data, 3 and 4 are real recordings, 9 is extra simulated with rare classes

        else:
            log_string('ERROR: Unknown dataset splits')
            exit()
    for split_cnt, split in enumerate(test_splits):
        log_string('\n\n---------------------------------------------------------------------------------------------------')
        log_string('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        log_string('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        loc_feat = params['dataset']
        if params['dataset'] == 'mic':
            if params['use_salsalite']:
                loc_feat = '{}_salsa'.format(params['dataset'])
            else:
                loc_feat = '{}_gcc'.format(params['dataset'])
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_split{}_{}_{}'.format(
            task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
        )
        model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
        model_name_final = '{}_model_final.h5'.format(os.path.join(params['model_dir'], unique_name))
        log_string("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        log_string('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        log_string('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True
        )

        # Collect i/o data size and load model configuration
        if params['model'] == 'seldnet':
            if params['modality'] == 'audio_visual':
                data_in, vid_data_in, data_out = data_gen_train.get_data_sizes()
                model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in).to(device)
            else:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = seldnet_model.SeldModel(data_in, data_out, params).to(device)

        elif params['model'] == 'myseldnet':
            if params['modality'] == 'audio_visual':
                data_in, vid_data_in, data_out = data_gen_train.get_data_sizes()
                model = seldnet_model.SeldModel(data_in, data_out, params, vid_data_in).to(device)
            else:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = seldnet_model.MySeldModel(data_in, data_out, params).to(device)

        elif params['model'] == 'ngccmodel':
            if params['modality'] == 'audio_visual':
                data_in, vid_data_in, data_out = data_gen_train.get_data_sizes()
                model = NGCCModel(data_in, data_out, params, vid_data_in).to(device)
            else:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = NGCCModel(data_in, data_out, params).to(device)

        elif params['model'] == 'cstformer':
            if params['modality'] == 'audio_visual':
                data_in, vid_data_in, data_out = data_gen_train.get_data_sizes()
                model = CST_former(data_in, data_out, params, vid_data_in).to(device)
            else:
                data_in, data_out = data_gen_train.get_data_sizes()
                model = CST_former(data_in, data_out, params).to(device)



                
        else:
            print('ERROR: Unknown model configuration')
            exit()

        if params['finetune_mode']:
            log_string('Running in finetuning mode. Initializing the model to the weights - {}'.format(params['pretrained_model_weights']))
            state_dict = torch.load(params['pretrained_model_weights'], map_location='cpu')
            if params['modality'] == 'audio_visual':
                state_dict = {k: v for k, v in state_dict.items() if 'fnn' not in k}
            if params['model'] == 'ngccmodel':
                # skip layers with non-matching shapes when loading weights
                model_dict = model.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
            model.load_state_dict(state_dict, strict=False)


        log_string('---------------- SELD-net -------------------')
        log_string('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
        log_string('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n, rnn_size: {}\n, nb_attention_blocks: {}\n, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'], params['nb_self_attn_layers'],
            params['fnn_size']))
        if not params['predict_tdoa']:
            summary(model, data_in[1:])

        # Dump results in DCASE output format for calculating final scores
        dcase_output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        log_string('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder))

        if params['predict_tdoa']:
            criterion_tdoa = TdoaLoss(fs=params['fs'], max_tau=params['max_tau'], tracks=params['tracks'])# max_events=params['max_events'])
        else:
            criterion_tdoa = None
        

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err, best_rel_dist_err = 1., 0., 180., 0., 9999, 999999., 999999.
        patience_cnt = 0

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']

        model_parameters = [
            (name, p) for (name, p) in model.named_parameters() if not name.startswith('q')]
        no_decay = ['bias', 'norm', 'Norm', 'cls', 'pos']
        # Apply weight decay to all layers, except biases, normalization layers and and learnable tokens
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model_parameters if not any(
            nd in n for nd in no_decay)], 'weight_decay': params['weight_decay']},
        {'params': [p for n, p in model_parameters if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=params['lr'])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=nb_epoch, eta_min=params['final_lr'])
        if not params['predict_tdoa']:
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=params['warmup'], after_scheduler=scheduler)
        optimizer.zero_grad()
        optimizer.step()

        if params['multi_accdoa'] is True:
            criterion = seldnet_model.MSELoss_ADPIT(relative_dist=params['relative_dist'])
        else:
            criterion = nn.MSELoss()

        # initialize validation scores to nan
        val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        val_time = np.nan
        val_loss = np.nan

        for epoch_cnt in range(nb_epoch):
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            #device = torch.device('cuda')
            #model = model.to(device)
            train_loss = train_epoch(data_gen_train, optimizer, model, criterion, params, device, criterion_tdoa)
            scheduler.step()
            train_time = time.time() - start_time
            if params['predict_tdoa']:
                log_string("saving TDOA model")
                torch.save(model.state_dict(), model_name_final)
            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------

            if (epoch_cnt > 0 and epoch_cnt % params['eval_freq'] == 0) or epoch_cnt == 0 or epoch_cnt == nb_epoch-1:
                start_time = time.time()
                #device = torch.device('cpu')
                #model = model.to(device)
                val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device, criterion_tdoa)
                # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores

                val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder)

                val_time = time.time() - start_time

                # Save model if F-score is good
                if val_F >= best_F:
                    best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr, val_dist_err
                    best_rel_dist_err = val_rel_dist_err
                    torch.save(model.state_dict(), model_name)
                    patience_cnt = 0
                else:
                    patience_cnt += params['eval_freq']

                if epoch_cnt == nb_epoch - 1:
                    log_string("saving final model")
                    torch.save(model.state_dict(), model_name_final)

            # Print stats
            log_string(
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                'F/AE/Dist_err/Rel_dist_err/SELD: {}, '
                'best_val_epoch: {} {}'.format(
                    epoch_cnt, train_time, val_time,
                    train_loss, val_loss,
                    '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_scr),
                    best_val_epoch,
                    '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format( best_F, best_LE, best_dist_err, best_rel_dist_err, best_seld_scr))
            )

            if patience_cnt > params['patience']:
                break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        # don't load best model, this is cherry picking
        log_string('Not loading best model weights, using final model weights instead')
        #log_string('Load best model weights')
        #model.load_state_dict(torch.load(model_name, map_location='cpu'))

        log_string('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True,
        )

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        log_string('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))


        test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device, criterion_tdoa)

        use_jackknife=True
        test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )

        log_string('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
        log_string('SED metrics: F-score: {:0.1f} {}'.format(100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
        log_string('DOA metrics: Angular error: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else ''))
        log_string('Distance metrics: {:0.2f} {}'.format(test_dist_err[0] if use_jackknife else test_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_dist_err[1][0], test_dist_err[1][1]) if use_jackknife else ''))
        log_string('Relative Distance metrics: {:0.2f} {}'.format(test_rel_dist_err[0] if use_jackknife else test_rel_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_rel_dist_err[1][0], test_rel_dist_err[1][1]) if use_jackknife else ''))

        if params['average']=='macro':
            log_string('Classwise results on unseen test data')
            log_string('Class\tF\tAE\tdist_err\treldist_err\tSELD_score')
            for cls_cnt in range(params['unique_classes']):
                log_string('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                    cls_cnt,

                    classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else '',

                    classwise_test_scr[0][6][cls_cnt] if use_jackknife else classwise_test_scr[6][cls_cnt],
                    '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][6][cls_cnt][0],
                                                classwise_test_scr[1][6][cls_cnt][1]) if use_jackknife else ''))

    LOG_FOUT.close()
                    
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

