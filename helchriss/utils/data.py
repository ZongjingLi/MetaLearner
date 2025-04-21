'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-12-24 18:42:10
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-24 18:42:21
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn
import random
import itertools
from typing import List, Any
from helchriss.logger import get_logger

logger = get_logger(__file__)
class BaseDataset(object):
    """An abstract class representing a Dataset."""
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        from torch.utils.data.dataset import ConcatDataset
        return ConcatDataset([self, other])

class IterableDatasetMixin(object):
    def __iter__(self):
        for i in range(len(self)):
            yield i, self[i]

class ListDataset(BaseDataset, IterableDatasetMixin):
    def __init__(self, data : List[Any]):
        """
        Args:
            data (list[Any]): the list of data.
        """
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return self.data[idx]

    def add(self, data): self.data.append(data)

class FilterableDatasetUnwrapped(BaseDataset, IterableDatasetMixin):
    """
    A filterable dataset. User can call various `filter_*` operations to obtain a subset of the dataset.
    """
    def __init__(self):
        super().__init__()
        self.metainfo_cache = dict()

    def get_metainfo(self, index):
        if index not in self.metainfo_cache:
            self.metainfo_cache[index] = self._get_metainfo(index)
        return self.metainfo_cache[index]

    def _get_metainfo(self, index):
        raise NotImplementedError()

class FilterableDatasetView(FilterableDatasetUnwrapped):
    def __init__(self, owner_dataset, indices=None, filter_name=None, filter_func=None):
        """
        Args:
            owner_dataset (Dataset): the original dataset.
            indices (List[int]): a list of indices that was filterred out.
            filter_name (str): human-friendly name for the filter.
            filter_func (Callable): just for tracking.
        """
        super().__init__()
        self.owner_dataset = owner_dataset
        self.indices = indices
        self._filter_name = filter_name
        self._filter_func = filter_func

    @property
    def unwrapped(self):
        if self.indices is not None:
            return self.owner_dataset.unwrapped
        return self.owner_dataset

    @property
    def filter_name(self):
        return self._filter_name if self._filter_name is not None else '<anonymous>'

    @property
    def full_filter_name(self):
        if self.indices is not None:
            return self.owner_dataset.full_filter_name + '/' + self.filter_name
        return '<original>'

    @property
    def filter_func(self):
        return self._filter_func

    def collect(self, key_func):
        return {key_func(self.get_metainfo(i)) for i in range(len(self))}

    def filter(self, filter_func, filter_name=None):
        indices = []
        for i in range(len(self)):
            metainfo = self.get_metainfo(i)
            if filter_func(metainfo):
                indices.append(i)
        if len(indices) == 0:
            raise ValueError('Filter results in an empty dataset.')
        logger.critical('Filter dataset {}: #before={}, #after={}.'.format(filter_name, len(self), len(indices)))
        return type(self)(self, indices, filter_name, filter_func)

    def random_trim_length(self, length):
        assert length < len(self)
        logger.info('Randomly trim the dataset: #samples = {}.'.format(length))
        indices = list(random.sample(range(len(self)), length))
        return type(self)(self, indices=indices, filter_name='randomtrim[{}]'.format(length))

    def trim_length(self, length):
        if type(length) is float and 0 < length <= 1:
            length = int(len(self) * length)
        assert length < len(self)
        logger.info('Trim the dataset: #samples = {}.'.format(length))
        return type(self)(self, indices=list(range(0, length)), filter_name='trim[{}]'.format(length))

    def trim_range(self, begin, end=None):
        if end is None:
            end = len(self)
        assert end <= len(self)
        logger.info('Trim the dataset: #samples = {}.'.format(end - begin))
        return type(self)(self, indices=list(range(begin, end)), filter_name='trimrange[{}:{}]'.format(begin, end))

    def split_trainval(self, split):
        if isinstance(split, float) and 0 < split < 1:
            split = int(len(self) * split)
        split = int(split)

        assert 0 < split < len(self)
        nr_train = split
        nr_val = len(self) - nr_train
        logger.info('Split the dataset: #training samples = {}, #validation samples = {}.'.format(nr_train, nr_val))
        return (
                type(self)(self, indices=list(range(0, split)), filter_name='train'),
                type(self)(self, indices=list(range(split, len(self))), filter_name='val')
        )

    def split_kfold(self, k):
        assert len(self) % k == 0
        block = len(self) // k

        for i in range(k):
            yield (
                    type(self)(self, indices=list(range(0, i * block)) + list(range((i + 1) * block, len(self))), filter_name='fold{}[train]'.format(i + 1)),
                    type(self)(self, indices=list(range(i * block, (i + 1) * block)), filter_name='fold{}[val]'.format(i + 1))
            )

    def repeat(self, nr_repeats):
        indices = list(itertools.chain(*[range(len(self)) for _ in range(nr_repeats)]))
        logger.critical('Repeat the dataset: #before={}, #after={}.'.format(len(self), len(indices)))
        return type(self)(self, indices=indices, filter_name='repeat[{}]'.format(nr_repeats))

    def sort(self, key, key_name=None):
        if key_name is None:
            key_name = str(key)
        indices = sorted(range(len(self)), key=lambda x: key(self.get_metainfo(x)))
        return type(self)(self, indices=indices, filter_name='sort[{}]'.format(key_name))

    def random_shuffle(self):
        indices = list(range(len(self)))
        random.shuffle(indices)
        return type(self)(self, indices=indices, filter_name='random_shuffle')

    def __getitem__(self, index):
        if self.indices is None:
            return self.owner_dataset[index]
        return self.owner_dataset[self.indices[index]]

    def __len__(self):
        if self.indices is None:
            return len(self.owner_dataset)
        return len(self.indices)

    def get_metainfo(self, index):
        if self.indices is None:
            return self.owner_dataset.get_metainfo(index)
        return self.owner_dataset.get_metainfo(self.indices[index])


# Create a custom dataset implementation with metadata
class ImageDataset(FilterableDatasetUnwrapped):  # Change the parent class
    def __init__(self, images, labels, categories=None):
        super().__init__()  # No owner dataset needed
        self.images = images
        self.labels = labels
        self.categories = categories or {}
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label
    
    def _get_metainfo(self, index):
        """Get metadata for an image"""
        category = self.categories.get(self.labels[index], "unknown")
        # Return metadata dictionary
        return {
            'index': index,
            'label': self.labels[index],
            'category': category,
            'shape': self.images[index].shape if hasattr(self.images[index], 'shape') else None
        }

from abc import abstractmethod


class GroundBaseDataset(BaseDataset):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __getitem__(self,idx):
        return 

class PlannerBaseDataset(BaseDataset):
    pass