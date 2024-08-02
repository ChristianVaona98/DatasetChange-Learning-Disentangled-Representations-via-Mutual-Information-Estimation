import os
import itertools
import random
import numpy as np
import torch

from utils.custom_typing import SmallNORBKey

PREFIXES = {
    'train': 'smallnorb-5x46789x9x18x6x2x96x96-training-',
    'test': 'smallnorb-5x01235x9x18x6x2x96x96-testing-',
}

FILE_TYPES = ['info', 'cat', 'dat']

SUFFIX = '.mat'

MAPPER = {
    '1e3d4c55': np.uint8,
    '1e3d4c54': np.int32,

    '1e3d4c51': np.float32,
    '1e3d4c53': np.float64,
    '1e3d4c56': np.int16,
}

ATTRIBUTE_TYPES = {
    'cat' : list(range(5)),
    'istances_train' : [4,6,7,8,9],
    'istances_test' : [0,1,2,3,5],
    'elevation' : list(range(9)),
    'azimuth' : list(range(0, 35, 2)),
    'lightning' : list(range(6)),
}

loaded_data = {}

# helper function to read int from file
def read_int(f):
    # Read 4 bytes from the file
    # Convert the bytes to an integer using little-endian byte order
    return int.from_bytes(f.read(4), byteorder='little')

def load_data_from_path(file_loc):
    with open(file_loc, 'rb') as f:
        # Read the magic_num, convert it to hexadecimal, and look up the data_type
        raw_magic_num = read_int(f)
        magic_num = format(raw_magic_num, '02x')
        data_type = MAPPER[magic_num]

        # Read how many dimensions to expect
        ndim = read_int(f)
        
        # Read at least 3 ints, or however many ndim there are
        shape = [
            read_int(f)
            for i in range(max(ndim, 3))
        ]   
        # But in case ndims < 3, take at most n_dim elements
        shape = shape[:ndim]

        return np.fromfile(
            f, 
            dtype=data_type, 
            count=np.prod(shape)
        ).reshape(shape)
 
def load_dataset(split='train', datafolder ="data/SmallNORB"):
    result = {}
    prefix = PREFIXES[split]
    for filetype in FILE_TYPES:
        filename = prefix + filetype + SUFFIX
        print('Reading {}'.format(filename))
        
        file_loc = os.path.join(datafolder, filename)

        result[(split, filetype)] = torch.tensor(load_data_from_path(file_loc), dtype=torch.float32)
    
    return result

def create_smallnorb_seeker(data: dict, dataset_type: str):
    # Map infos/category to the index of the image
    data_lookup = {
        SmallNORBKey(category.item(), *(info.tolist())): i
        for i, (info, category) in enumerate(zip(data[(dataset_type, "info")], data[(dataset_type, "cat")]))
    }
    return data_lookup

def generate_imgs_pair_attributes(combinations:int, split:str='train'):
    all_couples = change_two_attributes(
        ATTRIBUTE_TYPES['cat'],
        ATTRIBUTE_TYPES[f'istances_{split}'],
        ATTRIBUTE_TYPES['elevation'],
        ATTRIBUTE_TYPES['azimuth'],
        ATTRIBUTE_TYPES['lightning'],
        combinations,
    )

    # find the duplicate lists
    all_couples.sort()
    all_couples = list(k for k,_ in itertools.groupby(all_couples))
    random.shuffle(all_couples)

    return all_couples

def change_two_attributes(cat, istances, elevation, azimuth, lightning, combinations:int):
    valid_column_changable = {
        0:'cat', 
        2:'elevation',
        3:'azimuth',
        4:'lightning'
    }

    c_elevation = 2
    c_lightning = 4

    all_combinations = list(itertools.product(cat, istances, [1], azimuth, [1]))

    default_elevation = elevation[0]
    default_azimuth = azimuth[0]
    default_lightning = lightning[0]

    cond = []
    already_seen_first = set()
    for _ in range(combinations):
        # left img change elevation
        # right img change lightning

        left_img = list(random.choice(all_combinations))
        left_img[c_elevation] = random.choice(ATTRIBUTE_TYPES[valid_column_changable[c_elevation]])
        left_img[c_lightning] = default_lightning
        
        while tuple(left_img) in already_seen_first:
            left_img = list(random.choice(all_combinations))
            left_img[c_elevation] = random.choice(ATTRIBUTE_TYPES[valid_column_changable[c_elevation]])
        
        right_img = [*left_img]
        right_img[c_elevation] = default_elevation

        right_img[c_lightning] = random.choice(
            list( set(ATTRIBUTE_TYPES[valid_column_changable[c_lightning]]) - set([default_lightning]) )
        )

        cond.append([[*left_img], [*right_img]])
    
    return cond