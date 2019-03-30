import torch
import torch.utils.data as data
from torchvision.datasets.folder import  has_file_allowed_extension, is_image_file, IMG_EXTENSIONS, pil_loader, accimage_loader,default_loader

from PIL import Image

import sys
import os
import os.path
import numpy as np
from random import shuffle


REGIONS_DICT={'Alabama': 'South', 'Arizona': 'SW',
 'California': 'Pacific',
 'Florida': 'South',
 'Indiana': 'MW',
 'Iowa': 'MW',
 'Kansas': 'MW',
 'Massachusetts': 'NE',
 'Michigan': 'MW',
 'Missouri': 'South',
 'Montana': 'RM',
 'New-York': 'MA',
 'North-Carolina': 'South',
 'Ohio': 'MW',
 'Oklahoma': 'SW',
 'Oregon': 'Pacific',
 'Pennsylvania': 'MA',
 'South-Carolina': 'South',
 'South-Dakota': 'MW',
 'Texas': 'SW',
 'Utah': 'RM',
 'Vermont': 'NE',
 'Virginia': 'South',
 'Washington': 'Pacific',
 'Wyoming': 'RM'}

REGIONS_TO_IDX={'RM': 6,'MA': 1,'NE': 2,'South': 3, 'Pacific': 4, 'MW': 0 , 'SW': 5}
IDX_TO_REGIONS={ 6:'RM',1:'MA',2:'NE',3:'South',4: 'Pacific', 0:'MW', 5:'SW'}


def make_dataset(dir, class_to_idx, extensions, domains):
    images = []
    meta = []

    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            available_regions = []
            available_years = []
            for domain in domains:
                year,place=domain
                if not place in available_regions:
                    available_regions.append(place)
                if not int(year) in available_years:
                    available_years.append(int(year))

            available_years = np.array(available_years)
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    year=int(path.split('/')[-1].split('_')[0])
                    city=(path.split('/')[-1].split('_')[1])
                    region=REGIONS_DICT[city]
                    delta = (year - available_years)
                    validity = (delta<10)*(delta>=0)
                    pivot_index = np.nonzero(validity)
                    try:
                        pivot_year = available_years[pivot_index].item()
                        if region in available_regions:
                            item = (path, class_to_idx[target])
                            images.append(item)
                            meta.append([year,region])
                    except:
                        continue

    return images, meta



class Portraits(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,domains=[]):
        extensions = IMG_EXTENSIONS
        loader = default_loader

        classes, class_to_idx = self._find_classes(root)
        samples, self.meta = make_dataset(root, class_to_idx, extensions, domains)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples


    def _find_classes(self, dir):

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        y,p=self.meta[index]

        return sample, (y, p), target

    def get_meta(self):
        return np.array(self.meta)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class PortraitsSampler(torch.utils.data.sampler.Sampler):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source, bs):
        self.data_source=data_source
        self.meta=np.array(self.data_source.get_meta())
        self.dict_meta={}
        self.indeces={}
        self.keys=[]
        self.bs=bs
        for idx, (year,region) in enumerate(self.meta):
            try:
                self.dict_meta[str(year*20+region)].append(idx)
            except:
                self.dict_meta[str(year*20+region)]=[idx]
                self.keys.append(str(year*20+region))
                self.indeces[str(year*20+region)]=0

        for idx in self.keys:
            shuffle(self.dict_meta[idx])

    def _sampling(self,idx, n):
        if self.indeces[idx]+n>=len(self.dict_meta[idx]):
            self.dict_meta[idx]=self.dict_meta[idx]+self.dict_meta[idx]
        self.indeces[idx]=self.indeces[idx]+n
        return self.dict_meta[idx][self.indeces[idx]-n:self.indeces[idx]]



    def _shuffle(self):
        order=np.random.randint(len(self.keys),size=(len(self.data_source)//(self.bs)))
        sIdx=[]
        for i in order:
            sIdx=sIdx+self._sampling(self.keys[i],self.bs)
        return np.array(sIdx)


    def __iter__(self):
        return iter(self._shuffle())

    def __len__(self):
        return len(self.data_source)/self.bs*self.bs
