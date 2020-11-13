# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Data consisting of multiple aligned parts.
"""
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union, Generic, Iterable, Iterator, Sequence, TypeVar)

import torch
from texar.torch.data.data.data_base import (
    DatasetBase, DataSource,
    FilterDataSource, ZipDataSource, SequenceDataSource)
from texar.torch.data.data.dataset_utils import Batch, connect_name
from texar.torch.data.data.mono_text_data import (
    MonoTextData, _LengthFilterMode, _default_mono_text_dataset_hparams)
from texar.torch.data.data.record_data import (
    PickleDataSource, RecordData, _default_record_dataset_hparams)
from texar.torch.data.data.scalar_data import (
    ScalarData, _default_scalar_dataset_hparams)
from texar.torch.data.data.text_data_base import (
    TextDataBase, TextLineDataSource)
from texar.torch.data.data.multi_aligned_data import (MultiAlignedData)
from texar.torch.data.embedding import Embedding
from texar.torch.data.vocabulary import SpecialTokens, Vocab
from texar.torch.hyperparams import HParams
from texar.torch.utils import utils, dict_fetch
from texar.torch.utils.dtypes import is_str, get_supported_scalar_types

import argparse
import functools
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from texar.torch.run import *
#from texar_pytorch.texar.torch.run import *
import math
import torch
from torch import nn
import texar.torch as tx
#import texar_pytorch.texar.torch as tx

from model import Transformer
import utils

__all__ = [
    "_default_dataset_hparams",
    "MultiAlignedData"
]

RawExample = TypeVar('RawExample')  # type of a raw example loaded from source
Example = TypeVar('Example')  # type of a data example

def _is_text_data(data_type):
    return data_type == "text"


def _is_scalar_data(data_type):
    return data_type in get_supported_scalar_types()


def _is_record_data(data_type):
    return data_type == "record"


def _default_dataset_hparams(data_type=None):
    r"""Returns hyperparameters of a dataset with default values.

    See :meth:`texar.torch.data.MultiAlignedData.default_hparams` for details.
    """
    if data_type is None:
        data_type = "text"

    if _is_text_data(data_type):
        hparams = _default_mono_text_dataset_hparams()
        hparams.update({
            "data_type": data_type,
            "vocab_share_with": None,
            "embedding_init_share_with": None,
            "processing_share_with": None,
        })
    elif _is_scalar_data(data_type):
        hparams = _default_scalar_dataset_hparams()
    elif _is_record_data(data_type):
        hparams = _default_record_dataset_hparams()
        hparams.update({
            "data_type": data_type,
        })
    else:
        raise ValueError(f"Invalid data type '{data_type}'")
    return hparams


class MultiAlignedDataMultiFiles(MultiAlignedData):
    r"""Data consisting of multiple aligned parts.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
        device: The device of the produced batches. For GPU training, set to
            current CUDA device.

    The processor can read any number of parallel fields as specified in
    the "datasets" list of :attr:`hparams`, and result in a Dataset whose
    element is a python `dict` containing data fields from each of the
    specified datasets. Fields from a text dataset or Record dataset have
    names prefixed by its :attr:`"data_name"`. Fields from a scalar dataset are
    specified by its :attr:`"data_name"`.

    Example:

        .. code-block:: python

            hparams={
                'datasets': [
                    {'files': 'a.txt', 'vocab_file': 'v.a', 'data_name': 'x'},
                    {'files': 'b.txt', 'vocab_file': 'v.b', 'data_name': 'y'},
                    {'files': 'c.txt', 'data_type': 'int', 'data_name': 'z'}
                ]
                'batch_size': 1
            }
            data = MultiAlignedData(hparams)
            iterator = DataIterator(data)

            for batch in iterator:
                # batch contains the following
                # batch == {
                #    'x_text': [['<BOS>', 'x', 'sequence', '<EOS>']],
                #    'x_text_ids': [['1', '5', '10', '2']],
                #    'x_length': [4]
                #    'y_text': [['<BOS>', 'y', 'sequence', '1', '<EOS>']],
                #    'y_text_ids': [['1', '6', '10', '20', '2']],
                #    'y_length': [5],
                #    'z': [1000],
                # }

            ...

            hparams={
                'datasets': [
                    {'files': 'd.txt', 'vocab_file': 'v.d', 'data_name': 'm'},
                    {
                        'files': 'd.tfrecord',
                        'data_type': 'tf_record',
                        "feature_types": {
                            'image': ['tf.string', 'stacked_tensor']
                        },
                        'image_options': {
                            'image_feature_name': 'image',
                            'resize_height': 512,
                            'resize_width': 512,
                        },
                        'data_name': 't',
                    }
                ]
                'batch_size': 1
            }
            data = MultiAlignedData(hparams)
            iterator = DataIterator(data)
            for batch in iterator:
                # batch contains the following
                # batch_ == {
                #    'x_text': [['<BOS>', 'NewYork', 'City', 'Map', '<EOS>']],
                #    'x_text_ids': [['1', '100', '80', '65', '2']],
                #    'x_length': [5],
                #
                #    # "t_image" is a list of a "numpy.ndarray" image
                #    # in this example. Its width is equal to 512 and
                #    # its height is equal to 512.
                #    't_image': [...]
                # }

    """

    def __init__(self, hparams, device: Optional[torch.device] = None):
        print("Using local texar")
        self._hparams = HParams(hparams, self.default_hparams())
        # Defaultizes hyperparameters of each dataset
        datasets_hparams = self._hparams.datasets
        defaultized_datasets_hparams = []
        for hparams_i in datasets_hparams:
            data_type = hparams_i.get("data_type", None)
            #print("data_type:", data_type)
            defaultized_ds_hpms = HParams(hparams_i,
                                          _default_dataset_hparams(data_type))
            defaultized_datasets_hparams.append(defaultized_ds_hpms)
        self._hparams.datasets = defaultized_datasets_hparams

        #print("will make_vocab")
        self._vocab = self.make_vocab(self._hparams.datasets)
        #print("will make_embedding")
        self._embedding = self.make_embedding(
            self._hparams.datasets, self._vocab)

        dummy_source = SequenceDataSource[Any]([])
        name_prefix: List[str] = []
        self._names: List[Dict[str, Any]] = []
        sources: List[DataSource] = []
        filters: List[Optional[Callable[[str], bool]]] = []
        self._databases: List[DatasetBase] = []
        for idx, hparams_i in enumerate(self._hparams.datasets):
            data_type = hparams_i.data_type
            source_i: DataSource

            if _is_text_data(data_type):
                #print("will TextLineDataSource")
                source_i = TextLineDataSource(
                    hparams_i.files,
                    compression_type=hparams_i.compression_type,
                    delimiter=hparams_i.delimiter)
                sources.append(source_i)
                if ((hparams_i.length_filter_mode ==
                     _LengthFilterMode.DISCARD.value) and
                        hparams_i.max_seq_length is not None):

                    def _get_filter(max_seq_length):
                        return lambda x: len(x) <= max_seq_length

                    filters.append(_get_filter(hparams_i.max_seq_length))
                else:
                    filters.append(None)

                self._names.append({
                    field: connect_name(hparams_i.data_name, field)
                    for field in ["text", "text_ids", "length"]
                })

                dataset_hparams = dict_fetch(
                    hparams_i, MonoTextData.default_hparams()["dataset"])
                dataset_hparams["data_name"] = None
                self._databases.append(MonoTextData(
                    hparams={"dataset": dataset_hparams}, device=device,
                    vocab=self._vocab[idx],
                    embedding=self._embedding[idx],
                    data_source=dummy_source))
            elif _is_scalar_data(data_type):
                source_i = TextLineDataSource(
                    hparams_i.files,
                    compression_type=hparams_i.compression_type)
                sources.append(source_i)
                filters.append(None)
                self._names.append({"data": hparams_i.data_name})

                dataset_hparams = dict_fetch(
                    hparams_i, ScalarData.default_hparams()["dataset"])
                dataset_hparams["data_name"] = "data"
                self._databases.append(ScalarData(
                    hparams={"dataset": dataset_hparams}, device=device,
                    data_source=dummy_source))
            elif _is_record_data(data_type):
                source_i = PickleDataSource(file_paths=hparams_i.files)
                sources.append(source_i)
                # TODO: Only check `feature_types` when we finally remove
                #   `feature_original_types`.
                feature_types = (hparams_i.feature_types or
                                 hparams_i.feature_original_types)
                self._names.append({
                    name: connect_name(hparams_i.data_name, name)
                    for name in feature_types.keys()})
                filters.append(None)

                dataset_hparams = dict_fetch(
                    hparams_i, RecordData.default_hparams()["dataset"])
                self._databases.append(RecordData(
                    hparams={"dataset": dataset_hparams}, device=device,
                    data_source=dummy_source))
            else:
                raise ValueError(f"Unknown data type: {hparams_i.data_type}")

            # check for duplicate names
            for i in range(1, len(name_prefix)):
                if name_prefix[i] in name_prefix[:i - 1]:
                    raise ValueError(f"Duplicate data name: {name_prefix[i]}")

            name_prefix.append(hparams_i["data_name"])

        self._name_to_id = {v: k for k, v in enumerate(name_prefix)}
        self._processed_cache = []
        self._datafile_id = 0 # for training from multiple files
        self._index_at_beginning_of_this_dataset = 0
        self._datafile_prefix = hparams_i.files
        #self._datafile_num = 33 # hparams_i.datafile_num
        #self._datafile_num = 64 # hparams_i.datafile_num
        #self._datafile_num = 3 # hparams_i.datafile_num
        #self._datafile_num = 16 # hparams_i.datafile_num
        #self._datafile_num = 26 # hparams_i.datafile_num
        self._datafile_num = 1 # hparams_i.datafile_num
        #self._datafile_num = 3 # hparams_i.datafile_num

        data_source: DataSource = ZipDataSource(*sources)

        if any(filters):
            def filter_fn(data):
                return all(fn(data) for fn, data in zip(filters, data)
                           if fn is not None)

            data_source = FilterDataSource(data_source, filter_fn=filter_fn)
        #print("data init derive done")
        super(MultiAlignedData, self).__init__(data_source, self._hparams, device)
        #self._dataset_size = 3000000
        #self._dataset_size = 6400000
        #self._dataset_size = 16000000
        #self._dataset_size = 3802215
        #self._dataset_size = 1250000
        #self._dataset_size = 3000
        self._dataset_size = 834229
        #self._dataset_size = 1262869
        #self._dataset_size = 27942549
        #self._dataset_size = hparams["total_sample_num_of_all_dataset"]
        #print("data init super done")
        #self.load_next_datafile()
 
    def load_next_one_datafile(self, file, dataset_index):
        f = open(file, "r")
        print("load file:", file, "by MultiAlignedData.load_next_one_datafile")
        res = [self._databases[dataset_index].process(line.strip().split()) for line in f]
        f.close()
        return res

    def load_next_datafile(self):
        #if self._datafile_id == -1: # means init
        #    self._datafile_id = 0
        src_filename = self._datafile_prefix + ".src"  + str(self._datafile_id)
        tgt_filename = self._datafile_prefix + ".tgt"  + str(self._datafile_id)
        #print("_datafile_prefix:", self._datafile_prefix)
        src_data = self.load_next_one_datafile(src_filename, 0)
        tgt_data = self.load_next_one_datafile(tgt_filename, 1)
        if len(src_data) != len(tgt_data):
            print("Error. len(src_data):", len(src_data), "!=len(tgt_data):", len(tgt_data))
        self._processed_cache = [(src, tgt) for src, tgt in zip(src_data, tgt_data)]
        self._datafile_id += 1
        if self._datafile_id >= self._datafile_num:
            self._datafile_id = 0
        #self._dataset_size = len(self._processed_cache)

        return

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters:

        .. code-block:: python

            {
                # (1) Hyperparams specific to text dataset
                "datasets": []
                # (2) General hyperparams
                "num_epochs": 1,
                "batch_size": 64,
                "allow_smaller_final_batch": True,
                "shuffle": True,
                "shuffle_buffer_size": None,
                "shard_and_shuffle": False,
                "num_parallel_calls": 1,
                "prefetch_buffer_size": 0,
                "max_dataset_size": -1,
                "seed": None,
                "name": "multi_aligned_data",
            }

        Here:

        1. "datasets" is a list of `dict` each of which specifies a
           dataset which can be text, scalar or Record. The :attr:`"data_name"`
           field of each dataset is used as the name prefix of the data fields
           from the respective dataset. The :attr:`"data_name"` field of each
           dataset should not be the same.

           i) For scalar dataset, the allowed hyperparameters and default
              values are the same as the "dataset" field of
              :meth:`texar.torch.data.ScalarData.default_hparams`. Note that
              :attr:`"data_type"` must be explicitly specified
              (either "int" or "float").

           ii) For Record dataset, the allowed hyperparameters and default
               values are the same as the "dataset" field of
               :meth:`texar.torch.data.RecordData.default_hparams`. Note that
               :attr:`"data_type"` must be explicitly specified ("record").

           iii) For text dataset, the allowed hyperparameters and default
                values are the same as the "dataset" filed of
                :meth:`texar.torch.data.MonoTextData.default_hparams`, with
                several extra hyperparameters:

                `"data_type"`: str
                    The type of the dataset, one of {"text", "int", "float",
                    "record"}. If set to "int" or "float", the dataset is
                    considered to be a scalar dataset. If set to
                    "record", the dataset is considered to be a Record
                    dataset.

                    If not specified or set to "text", the dataset is
                    considered to be a text dataset.

                `"vocab_share_with"`: int, optional
                    Share the vocabulary of a preceding text dataset with
                    the specified index in the list (starting from 0). The
                    specified dataset must be a text dataset, and must have
                    an index smaller than the current dataset.

                    If specified, the vocab file of current dataset is
                    ignored. Default is `None` which disables the vocab
                    sharing.

                `"embedding_init_share_with"`: int, optional
                    Share the embedding initial value of a preceding text
                    dataset with the specified index in the list (starting
                    from 0). The specified dataset must be a text dataset,
                    and must have an index smaller than the current dataset.

                    If specified, the :attr:`"embedding_init"` field of the
                    current dataset is ignored. Default is `None` which
                    disables the initial value sharing.

                `"processing_share_with"`: int, optional
                    Share the processing configurations of a preceding text
                    dataset with the specified index in the list (starting
                    from 0). The specified dataset must be a text dataset,
                    and must have an index smaller than the current dataset.

                    If specified, relevant field of the current dataset are
                    ignored, including `delimiter`, `bos_token`,
                    `eos_token`, and "other_transformations". Default is
                    `None` which disables the processing sharing.

        2. For the **general** hyperparameters, see
        :meth:`texar.torch.data.DatasetBase.default_hparams` for details.

        """
        hparams = TextDataBase.default_hparams()
        hparams["name"] = "multi_aligned_data"
        hparams["datasets"] = []
        hparams["shuffle"] = False # added by tzl for file by file loading
        hparams["total_sample_num_of_all_dataset"] = 0
        return hparams

    def __getitem__(self, index: Union[int, Tuple[int, RawExample]]) -> Example:
        #print("__getitem__ in multi aglin data. index:", index)
        if index % 5000 == 0:
            print("read", index, "samples by multi aglin data.")
        #print("read", index, "samples by multi aglin data.")
        #if self._datafile_id == -1: # means init
        #    self._processed_cache = []
        #    self._datafile_id = 0
        if index == 0 or index - self._index_at_beginning_of_this_dataset >= len(self._processed_cache):
            self.load_next_datafile()
            self._index_at_beginning_of_this_dataset = index
        if isinstance(index, int):
            #print("self._fully_cached:", self._fully_cached, "self._parallelize_processing:", self._parallelize_processing)
            if self._fully_cached:
                index_this = index - self._index_at_beginning_of_this_dataset
                #print("_processed_cache index:", index, "index_this:", index_this, \
                #    "_index_at_beginning_of_this_dataset:", self._index_at_beginning_of_this_dataset, \
                #    "data:", self._processed_cache[index_this])
                return self._processed_cache[index_this]
            elif not self._parallelize_processing:
                return self._transformed_source[index]
            else:
                return self.process(self._source[index])
        else:
            # `index` is a tuple of (index, example).
            if not self._parallelize_processing:
                return index[1]  # type: ignore
            else:
                return self.process(index[1])
