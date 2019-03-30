import torch
import torch.utils.data as data
from torchvision.datasets.folder import  has_file_allowed_extension, is_image_file, IMG_EXTENSIONS, pil_loader, accimage_loader,default_loader
import os
import os.path
import sys
import numpy as np
from PIL import Image

from random import shuffle

#### Here we filter the dataset according to the given metadata
def make_dataset(list_file, class_to_idx, extensions, domains):
    images = []
    meta=[]
    with open(list_file) as f:
    	lines = f.readlines()
    for l in lines:
        fname, domain, target = l.strip().split(' ')
        year,viewpoint=domain.split('-')
        if has_file_allowed_extension(fname, extensions) and (year,viewpoint) in domains:
                    path = fname
                    item = (path, class_to_idx[target])
                    meta.append([int(year),int(viewpoint)])
                    images.append(item)

    return images, meta


#### Here we define a dataset which takes as input files list of files
class Compcars(data.Dataset):

    def __init__(self, list_file, classes, transform=None, target_transform=None, domains=[]):
        extensions = IMG_EXTENSIONS
        loader = default_loader
        self.classes = classes
        self.class_to_idx = {'1':0,'2':1,'3':2,'4':3}
        samples, self.meta = make_dataset(list_file, self.class_to_idx, extensions, domains=domains)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in: " + list_file + "\n"))

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples


    def get_meta(self):
        return self.meta


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        meta=self.meta[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, (meta[0], meta[1]), target

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


#### Here we define a Sampler that has all the samples of each batch from the same domain
class CompcarsSampler(torch.utils.data.sampler.Sampler):
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
        for idx, (y,v) in enumerate(self.meta):
            try:
                self.dict_meta[str(y)+"-"+str(v)].append(idx)
            except:
                self.dict_meta[str(y)+"-"+str(v)]=[idx]
                self.keys.append(str(y)+"-"+str(v))
                self.indeces[str(y)+"-"+str(v)]=0
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
        return len(self.data_source)//self.bs*self.bs
