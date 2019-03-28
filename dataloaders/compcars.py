import torch
import torch.utils.data as data
import os
import os.path
import sys
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True




def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


#### Qui decidi cosa mettere dei file che trovi nei folder (o lista, come nel mio caso). 
#### Mi servivano i metadati, quindi ho ottenuto anche altre info in times
def make_dataset(list_file, class_to_idx, extensions, domains):
    images = []
    times=[]
    with open(list_file) as f:
    	lines = f.readlines()
    for l in lines:
	fname, meta, target = l.strip().split(' ')
    	if has_file_allowed_extension(fname, extensions) and meta in domains:
                    path = fname
                    item = (path, class_to_idx[target])
		    y,v=meta.split('-')
                    times.append([int(y),int(v)])
                    images.append(item)

    return images, times


#### Qui definisci il dataset e i metodi collegati, piu' associazione label indici e cavolate simili
class DatasetList(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, list_file, classes, loader, extensions, transform=None, target_transform=None, domains=[]):
        self.classes = classes
        self.class_to_idx = {'1':0,'2':1,'3':2, '4':3}
        samples, self.times = make_dataset(list_file, self.class_to_idx, extensions, domains=domains)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in: " + list_file + "\n"))

        self.loader = loader
        self.extensions = extensions


        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def get_times(self):
        return self.times

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
	meta=self.times[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, meta[0]-2009, meta[1]-1, target

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


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



#### Questo e' un sampler randomico che riempiva ogni batch con elementi dello stesso metadato
class CarRegressionSampler(torch.utils.data.sampler.Sampler):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source, bs):
        self.data_source=data_source
        self.time=np.array(self.data_source.get_times())
	self.dict_times={}
	self.indeces={}
	self.keys=[]
	self.bs=bs
	for idx, (y,v) in enumerate(self.time):
		try:
			self.dict_times[str(y)+"-"+str(v)].append(idx)
		except:
			self.dict_times[str(y)+"-"+str(v)]=[idx]
			self.keys.append(str(y)+"-"+str(v))
			self.indeces[str(y)+"-"+str(v)]=0

	for idx in self.keys:
		shuffle(self.dict_times[idx])

    def _sampling(self,idx, n):
	if self.indeces[idx]+n>=len(self.dict_times[idx]):
		self.dict_times[idx]=self.dict_times[idx]+self.dict_times[idx]
	self.indeces[idx]=self.indeces[idx]+n
	return self.dict_times[idx][self.indeces[idx]-n:self.indeces[idx]]

		

    def _shuffle(self):
	order=np.random.randint(len(self.keys),size=(len(self.data_source)/(self.bs)))	
	sIdx=[]
	for i in order:
		sIdx=sIdx+self._sampling(self.keys[i],self.bs)
        return np.array(sIdx)
        

    def __iter__(self):
        return iter(self._shuffle())

    def __len__(self):
        return len(self.data_source)/self.bs*self.bs


#### Qui e' il wrapper di tutto, forse non necessario al 100%, credo si possa usare direttamente l'a;tro
class FilteredList(DatasetList):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, list_file, classes, transform=None, target_transform=None,
                 loader=default_loader, domains=[]):
        super(FilteredList, self).__init__(list_file, classes, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform, domains=domains)
        self.imgs = self.samples

