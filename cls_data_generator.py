#
# Data generator for training the SELDnet
#
import torch
import torchvision.transforms as transforms
import os
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random
from PIL import Image
import matplotlib.pyplot as plt


class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()
        self._multi_accdoa = params['multi_accdoa']
        self.train_video = params['train_on_video']
        self.scale_down = params['scale_down']
        # if self._per_file:
        #    self.train_video = False

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._nb_classes = self._feat_cls.get_nb_classes()
        self._label_size = None
        self._vid_frame_size = None

        self._circ_buf_feat = None
        self._circ_buf_label = None
        self._circ_buf_vid_frame = None

        self._modality = params['modality']
        if self._modality == 'audio_visual':
            self._vid_feature_seq_len = self._label_seq_len  # video feat also at 10 fps same as label resolutions (100ms)
            self._vid_feat_dir = self._feat_cls.get_vid_feat_dir()
            self._circ_buf_vid_feat = None
            if self.train_video:
                self._vid_frame_dir = self._vid_feat_dir + '_frames'
                self._circ_buf_vid_frame = None

        self._get_filenames_list_and_feat_label_sizes()

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
        if self._is_eval:
            label_shape = None
        else:
            if self._multi_accdoa is True:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3*4)
            else:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*4)

        if self._modality == 'audio_visual':
            vid_feat_shape = (self._batch_size, self._vid_feature_seq_len, 7, 7)
            return feat_shape, vid_feat_shape, label_shape
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        print('Computing some stats about the dataset')
        max_frames, total_frames, temp_feat = -1, 0, []
        for filename in os.listdir(self._feat_dir):
            if int(filename[4]) in self._splits:  # check which split the file belongs to
                if self._modality == 'audio' or (hasattr(self, '_vid_feat_dir') and os.path.exists(os.path.join(self._vid_feat_dir, filename))):   # some audio files do not have corresponding videos. Ignore them.
                    self._filenames_list.append(filename)
                    temp_feat = np.load(os.path.join(self._feat_dir, filename))
                    # print("temp_feat.shape: ", temp_feat.shape)
                    total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len))
                    if temp_feat.shape[0]>max_frames:
                        max_frames = temp_feat.shape[0]

        if len(temp_feat) != 0:
            self._nb_frames_file = max_frames if self._per_file else temp_feat.shape[0]
            self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins
        else:
            print('Loading features failed')
            exit()

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            if self._multi_accdoa is True:
                self._num_track_dummy = temp_label.shape[-3]
                self._num_axis = temp_label.shape[-2]
                self._num_class = temp_label.shape[-1]
            else:
                self._label_len = temp_label.shape[-1]
            self._doa_len = 3 # Cartesian

        if self._per_file:
            self._batch_size = int(np.ceil(max_frames/float(self._feature_seq_len)))
            print('\tWARNING: Resetting batch size to {}. To accommodate the inference of longest file of {} frames in a single batch'.format(self._batch_size, max_frames))
            self._nb_total_batches = len(self._filenames_list)
        else:
            print("total_frames, self._batch_size, self._feature_seq_len: ", total_frames, self._batch_size, self._feature_seq_len)
            self._nb_total_batches = int(np.floor(total_frames / (self._batch_size*self._feature_seq_len)))

        self._feature_batch_seq_len = self._batch_size*self._feature_seq_len
        self._label_batch_seq_len = self._batch_size*self._label_seq_len

        if self._modality == 'audio_visual':
            self._vid_feature_batch_seq_len = self._batch_size*self._vid_feature_seq_len
            print("Here is self._vid_feature_batch_seq_len: ", self._vid_feature_batch_seq_len)

        return

    def generate(self):
        """
        Generates batches of samples
        :return:
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()

        if self._modality == 'audio_visual':
            self._circ_buf_vid_feat = deque()
            if self.train_video:
                self._circ_buf_vid_frame = deque()

        file_cnt = 0
        if self._is_eval:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while (len(self._circ_buf_feat) < self._feature_batch_seq_len or (hasattr(self, '_circ_buf_vid_feat')
                                                                                  and hasattr(self, '_vid_feature_batch_seq_len') and len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len)):
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)

                    if self._modality == 'audio_visual':
                        temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, self._filenames_list[file_cnt]))
                        for vf_row_cnt, vf_row in enumerate(temp_vid_feat):
                            self._circ_buf_vid_feat.append(vf_row)
                        if self.train_video:
                            print("Loading frame!")
                            temp_frame = np.load(os.path.join(self._vid_frame_dir, self._filenames_list[file_cnt]))
                            for vf_row_cnt, vf_row in enumerate(temp_frame):
                                self._circ_buf_vid_frame.append(vf_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                        for row_cnt, row in enumerate(extra_feat):
                            self._circ_buf_feat.append(row)

                        if self._modality == 'audio_visual':
                            vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                            extra_vid_feat = np.ones((vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6

                            for vf_row_cnt, vf_row in enumerate(extra_vid_feat):
                                self._circ_buf_vid_feat.append(vf_row)

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))

                if self._modality == 'audio_visual':
                    vid_feat = np.zeros((self._vid_feature_batch_seq_len, 1024, 7, 7))
                    for v in range(self._vid_feature_batch_seq_len):
                        vid_feat[v, :, :] = self._circ_buf_vid_feat.popleft()
                    vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)

                    yield feat, vid_feat
                else:
                    yield feat

        else:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while (len(self._circ_buf_feat) < self._feature_batch_seq_len or (hasattr(self, '_circ_buf_vid_feat')
                                                                                  and hasattr(self, '_vid_feature_batch_seq_len') and len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len)):
                    try:
                        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                        temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))
                    except IndexError:
                        temp_feat = np.zeros((32, 640))
                        temp_label = np.zeros((32, 6, 5, 13))
                    if self._modality == 'audio_visual':
                        try:
                            temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, self._filenames_list[file_cnt]))
                        except IndexError:
                            temp_vid_feat = np.zeros((32, 1024, 7, 7))
                        if self.train_video:
                            try:
                                temp_frame = np.load(os.path.join(self._vid_frame_dir, self._filenames_list[file_cnt]))
                                if self.scale_down:
                                    resized_frames = np.zeros((temp_frame.shape[0], 90, 180, 3))
                                    for idx, frame in enumerate(temp_frame):
                                        resized_frame = np.resize(frame, (90, 180, 3))
                                        resized_frames[idx] = resized_frame
                                    temp_frame = resized_frames
                            except IndexError:
                                temp_frame = np.zeros((32, 180, 360, 3))
                                if self.scale_down:
                                    temp_frame = np.zeros((32, 90, 180, 3))

                    if not self._per_file:
                        # Inorder to support variable length features, and labels of different resolution.
                        # We remove all frames in features and labels matrix that are outside
                        # the multiple of self._label_seq_len and self._feature_seq_len. Further we do this only in training.
                        temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                        temp_mul = temp_label.shape[0] // self._label_seq_len
                        temp_feat = temp_feat[:temp_mul * self._feature_seq_len, :]
                        if self._modality == 'audio_visual':
                            temp_vid_feat = temp_vid_feat[:temp_mul * self._vid_feature_seq_len, :, :]
                            if self.train_video:
                                temp_frame = temp_frame[:temp_mul * self._vid_feature_seq_len, :, :]

                    for f_row in temp_feat:
                        self._circ_buf_feat.append(f_row)
                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row)

                    if self._modality == 'audio_visual':
                        for vf_row in temp_vid_feat:
                            self._circ_buf_vid_feat.append(vf_row)
                        if self.train_video:
                            for frame_row in temp_frame:
                                self._circ_buf_vid_frame.append(frame_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                        if self._modality == 'audio_visual':
                            vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                            extra_vid_feat = np.ones(
                                (vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6
                            if self.train_video:
                                frame_extra_frames = self._vid_feature_batch_seq_len - temp_frame.shape[0]
                                extra_frame = np.ones(
                                    (frame_extra_frames, temp_frame.shape[1], temp_frame.shape[2],
                                     temp_frame.shape[3])) * 1e-6
                                # print("extra_vid_feat: ", extra_vid_feat.shape)
                                # print("extra_frame: ", extra_frame.shape)

                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        if self._multi_accdoa is True:
                            extra_labels = np.zeros(
                                (label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                        else:
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                        for f_row in extra_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)
                        if self._modality == 'audio_visual':
                            for vf_row in extra_vid_feat:
                                self._circ_buf_vid_feat.append(vf_row)
                            if self.train_video:
                                for frame_row in extra_frame:
                                    self._circ_buf_vid_frame.append(frame_row)

                    file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                if self._modality == 'audio_visual':
                    vid_feat = np.zeros((self._vid_feature_batch_seq_len, 1024, 7, 7))
                    for v in range(self._vid_feature_batch_seq_len):
                        vid_feat[v, :, :] = self._circ_buf_vid_feat.popleft()
                    if self.train_video:
                        if self.scale_down:
                            vid_frame = np.zeros((self._vid_feature_batch_seq_len, 90, 180, 3))
                        else:
                            vid_frame = np.zeros((self._vid_feature_batch_seq_len, 180, 360, 3))
                        for v in range(self._vid_feature_batch_seq_len):
                            vid_frame[v, :, :, :] = self._circ_buf_vid_frame.popleft()

                if self._multi_accdoa is True:
                    label = np.zeros(
                        (self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                    for j in range(self._label_batch_seq_len):
                        label[j, :, :, :] = self._circ_buf_label.popleft()
                else:
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()

                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))
                if self._modality == 'audio_visual':
                    vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)
                    if self.train_video:
                        vid_frame = self._vid_feat_split_in_seqs(vid_frame, self._vid_feature_seq_len) #OBS vet ej om detta blir rätt?

                label = self._split_in_seqs(label, self._label_seq_len)
                if self._multi_accdoa is True:
                    pass
                else:
                    mask = label[:, :, :self._nb_classes]
                    mask = np.tile(mask, 4)
                    label = mask * label[:, :, self._nb_classes:]
                if self.train_video:
                    yield feat, vid_feat, vid_frame, label
                elif self._modality == 'audio_visual':
                    yield feat, vid_feat, label
                else:
                    yield feat, label

    def generate_video(self):
        if self._shuffle:
            random.shuffle(self._filenames_list)

        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()

        # Define transformations
        preprocess = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL image
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        if self._modality == 'audio_visual':
            self._circ_buf_vid_feat = deque()

        file_cnt = 75
        for i in range(self._nb_total_batches):
            # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
            # circular buffer. If not keep refilling it.

            # print("Here is len(self._filenames_list): ", len(self._filenames_list))
            # print("len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len) ", len(self._circ_buf_vid_feat), self._vid_feature_batch_seq_len)

            while len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len:
                # temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                # temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))
                print("File count, filenames_list, circ_buf_vid_feat, self._vid_feature_batch_seq_len: ", file_cnt,
                      self._filenames_list[file_cnt], len(self._circ_buf_vid_feat), self._vid_feature_batch_seq_len)
                try:
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))
                except IndexError:
                    print("IndexError: list index out of range occurred.")
                    print("self._label_dir, file_cnt, _filenames_list: ", self._label_dir, file_cnt, self._filenames_list)
                    print("len(self._circ_buf_vid_feat) ", len(self._circ_buf_vid_feat))
                    # Handle the error here, such as logging or other actions you want to take
                    break

                if self._modality == 'audio_visual':
                    temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, self._filenames_list[file_cnt]),
                                            allow_pickle=True)

                if not self._per_file:
                    # Inorder to support variable length features, and labels of different resolution.
                    # We remove all frames in features and labels matrix that are outside
                    # the multiple of self._label_seq_len and self._feature_seq_len. Further we do this only in training.
                    temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                    temp_mul = temp_label.shape[0] // self._label_seq_len
                    # temp_feat = temp_feat[:temp_mul * self._feature_seq_len, :]
                    if self._modality == 'audio_visual':
                        # temp_vid_feat = temp_vid_feat[:temp_mul * self._vid_feature_seq_len, :, :]
                        temp_vid_feat = temp_vid_feat[:temp_mul * self._vid_feature_seq_len]

                # for f_row in temp_feat:
                #    self._circ_buf_feat.append(f_row)
                for l_row in temp_label:
                    self._circ_buf_label.append(l_row)
                if self._modality == 'audio_visual':
                    for vf_row in temp_vid_feat:
                        self._circ_buf_vid_feat.append(vf_row)

                # If self._per_file is True, this returns the sequences belonging to a single audio recording
                if self._per_file:
                    # feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                    # extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                    if self._modality == 'audio_visual':
                        vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                        extra_vid_feat = np.ones(
                            (vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6

                    label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                    if self._multi_accdoa is True:
                        extra_labels = np.zeros(
                            (label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                    else:
                        extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                    for l_row in extra_labels:
                        self._circ_buf_label.append(l_row)
                    if self._modality == 'audio_visual':
                        for vf_row in extra_vid_feat:
                            self._circ_buf_vid_feat.append(vf_row)

                file_cnt = file_cnt + 1

            # print("Reading video features")
            if self._modality == 'audio_visual':
                vid_feat = np.zeros((self._vid_feature_batch_seq_len, 3, 224, 224))
                for v in range(self._vid_feature_batch_seq_len):
                    img = self._circ_buf_vid_feat.popleft()

                    # Apply transformations to your data
                    preprocessed_img = preprocess(img)

                    # img.show()
                    # input("Press Enter to continue...")

                    img_array = np.array(preprocessed_img)

                    # Compute the mean separately for each pixel position across RGB channels
                    # pixel_means = np.mean(img_array, axis=-1)  # Calculate mean across RGB channels
                    # Assign the mean values to the corresponding element of vid_feat
                    vid_feat[v, :, :, :] = img_array
            # print("Done with that")
            if self._multi_accdoa is True:
                label = np.zeros(
                    (self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                for j in range(self._label_batch_seq_len):
                    label[j, :, :, :] = self._circ_buf_label.popleft()
            else:
                label = np.zeros((self._label_batch_seq_len, self._label_len))
                for j in range(self._label_batch_seq_len):
                    label[j, :] = self._circ_buf_label.popleft()

            if self._modality == 'audio_visual':
                vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)

            label = self._split_in_seqs(label, self._label_seq_len)
            if self._multi_accdoa is True:
                pass
            else:
                mask = label[:, :, :self._nb_classes]
                mask = np.tile(mask, 4)
                label = mask * label[:, :, self._nb_classes:]
            if self._modality == 'audio_visual':
                yield vid_feat, label

    def _split_in_seqs(self, data, _seq_len): # data - 250*8, 7, 64 - 250
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    def _vid_feat_split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 4:  # Check if data has 4 dimensions
            num_frames, channels, height, width = data.shape
            if num_frames % _seq_len:
                # Trim the data to ensure it can be evenly split into sequences
                num_frames = num_frames - (num_frames % _seq_len)
                data = data[:num_frames]
            # Reshape the data into sequences of length _seq_len
            data = data.reshape(-1, _seq_len, channels, height, width)
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            else:
                data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions for video features: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()

    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)
