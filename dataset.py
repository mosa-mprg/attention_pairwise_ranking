import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from pprint import pprint as print
import sys

class VideoRecord(object):
    def __init__(self, row):
        #print(row)
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        numgoodframe = sum(os.path.isfile(os.path.join(self._data[0], name)) for name in os.listdir(self._data[0]))
        return numgoodframe

    @property
    def path2(self):
        return self._data[1]

    @property
    def num_frames2(self):
        numbadframe = sum(os.path.isfile(os.path.join(self._data[1], name)) for name in os.listdir(self._data[1]))
        return numbadframe

class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=True):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, directory, idx):
        #print(directory)
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):
        #video_record = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        #self.video_list = [video_record[:2],video_record[2:]]
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _sample_indices2(self, record):
        average_duration = (record.num_frames2 - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames2 > self.num_segments:
            offsets = np.sort(randint(record.num_frames2 - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1
    """ 
    def _get_val_indices(self, record):
        tick = (record.num_frames - self.new_length ) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def _get_val_indices2(self, record):
        tick = (record.num_frames2 - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length ) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def _get_test_indices2(self, record):
        tick = (record.num_frames2 - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1
    """
    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices2(self, record):
        if record.num_frames2 > self.num_segments + self.new_length - 1:
            tick = (record.num_frames2 - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / 12
        tan = 12/self.num_segments
        if tan == 2:
            offsets = np.array([int(tick / 2.0 + tick * (x*2)) for x in range(self.num_segments)])
        elif tan == 4:
            offsets = np.array([int(tick / 2.0 + tick * (x*4)) for x in range(self.num_segments)])
        else:
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def _get_test_indices2(self, record):
        tick = (record.num_frames2 - self.new_length + 1) / 12
        tan = 12/self.num_segments
        if tan == 2:
            offsets = np.array([int(tick / 2.0 + tick * (x*2)) for x in range(self.num_segments)])
        elif tan == 4:
            offsets = np.array([int(tick / 2.0 + tick * (x*4)) for x in range(self.num_segments)])
        else:
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def __getitem__(self, index):
        #records = self.video_list[index]
        #data = []
        #for record in records:
        #    if not self.test_mode:
        #        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        #    else:
        #        segment_indices = self._get_test_indices(record)
        #    data.append(self.get(record, segment_indices))
        #return data
        
        record = self.video_list[index]
        #print("{} num frame:{}".format(record.path, record.num_frames))
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            segment_indices2 = self._sample_indices2(record) if self.random_shift else self._get_val_indices2(record)
        else:
            segment_indices = self._get_test_indices(record)
            segment_indices2 = self._get_test_indices2(record)

        output1 = self.get1(record, segment_indices)
        output2 = self.get2(record, segment_indices2)

        #return output1,output2,path,path2,p,p2
        return output1,output2


    def get1(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                #print('{} {}'.format(record.path, p), stream=sys.stdout)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data


    def get2(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path2, p)
                #print('{} {}'.format(record.path2, p), stream=sys.stdout)
                images.extend(seg_imgs)
                if p < record.num_frames2:
                    p += 1

        process_data = self.transform(images)
        return process_data

    def __len__(self):
        return len(self.video_list)
