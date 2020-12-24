import shutil
import struct
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm

import pickle
import os.path

class TPMNDataset(torch.utils.data.Dataset):
    """
    TMPN Click-Through Rate Prediction Dataset

    :param dataset_path: tpmn train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold

    """

    def __init__(self, dataset_path=None, cache_path='.tpmn', rebuild_cache=False, min_threshold=4):
        self.new_dict = {}
        self.char_index = 1
        self.NUM_FEATS = 31 # sample 17, imbalance 23, june 31, june_sampling 19, june_sample(add for LSTM) 34
        self.min_threshold = min_threshold
        
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            
            self.__build_cache(dataset_path, cache_path)
            # pickle.dump(self.__build_cache(dataset_path, cache_path), open(cache_path, 'wb'))
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
        
        cache_file = 'tpmn.pkl'
        if os.path.isfile(cache_file) :
            self.field_dims = pickle.load(open(cache_file, 'rb'))
        else :
            with self.env.begin(write=False) as txn:
                self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)
            pickle.dump(self.field_dims, open(cache_file, 'wb'))
        ## loading new_dict     
        with open('new_dict.pkl', 'rb') as f :
            self.new_dict = pickle.load(f)
            
    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        
        #return np_array[1:], np_array[0], np_array[self.NUM_FEATS+1:]
        return np_array[1:self.NUM_FEATS+1], np_array[0], np_array[self.NUM_FEATS+1:] # feature, click, additional1, 2
        
    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        feat_mapper, defaults = self.__get_feat_mapper(path)
        
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32) 
            for i, fm in feat_mapper.items():
                field_dims[i-1] = len(fm) + 1 # use length dict, determine field_dims
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)
            ## new_dict            
            with open('new_dict.pkl', 'wb') as f :
                pickle.dump(self.new_dict, f)
                
            
        
    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create tpmn dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 1: 
                    continue
                for i in range(1, self.NUM_FEATS + 1): 
                    feat_cnts[i][values[i]] += 1 
                
        # random 방지, 코드 수정, count가 높은 순서대로 번호 부여
        feat_mapper = {i: [feat for feat, c in sorted(cnt.items(), key=lambda item:item[1], reverse=True) if c >= self.min_threshold] for i, cnt in feat_cnts.items()}
        feat_mapper2 = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper2.items()}
        
        return feat_mapper2, defaults # key feature num, value feature value + count

    
    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create tpmn dataset cache: setup lmdb')
                
            for line in pbar:
                values = line.rstrip('\n').split(',') 
                
                if len(values) != self.NUM_FEATS + 1: 
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32) 
                np_array[0] = int(values[0]) # 0 : click
                for i in range(1, self.NUM_FEATS + 1): 
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                
                ### Insert encoded ifa and ip numbers into lmdb
                char_list = list(values[2] + values[5] + values[13])
                
                for char in char_list:
                    if char not in self.new_dict :
                        self.new_dict[char] = self.char_index
                        self.char_index += 1
                max_appbundle = 149
                max_carrier = 50
                max_make = 38
                
                encoded_appbundle = np.array([self.new_dict[char] for char in list(values[2])] + (max_appbundle - len(list(values[2]))) * [0], dtype=np.uint32)
                encoded_carrier = np.array([self.new_dict[char] for char in list(values[5])] + (max_carrier - len(list(values[5]))) * [0], dtype=np.uint32)
                encoded_make = np.array([self.new_dict[char] for char in list(values[13])] + (max_make - len(list(values[13]))) * [0], dtype=np.uint32)
                np_array = np.concatenate([np_array, encoded_appbundle ,encoded_carrier, encoded_make]).astype(dtype=np.uint32)
                
                #if len(np_array) != 79: # click 1 + feature 23 + encoded ifa 40 + encoded ip 15 = 79
                #    print("Array size error")
                    
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes())) # unsiged int
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer