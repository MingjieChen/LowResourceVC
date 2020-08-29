from torch.utils import data
import torch
import os
import random
import glob
from os.path import join, basename, dirname, split
import numpy as np

# Below is the accent info for the used 10 speakers.
#spk2acc = {'262': 'Edinburgh', #F
#           '272': 'Edinburgh', #M
#           '229': 'SouthEngland', #F 
#           '232': 'SouthEngland', #M
#           '292': 'NorthernIrishBelfast', #M 
#           '293': 'NorthernIrishBelfast', #F 
#           '360': 'AmericanNewJersey', #M
#           '361': 'AmericanNewJersey', #F
#           '248': 'India', #F
#           '251': 'India'} #M
#min_length = 256   # Since we slice 256 frames from each utterance when training.
# Build a dict useful when we want to get one-hot representation of speakers.
#speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
#spk2idx = dict(zip(speakers, range(len(speakers))))

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class PairDataset(data.Dataset):
    '''dataset for training with pair samples input'''
    
    def __init__(self, data_dir, speakers, min_length = 256, few_shot = None):
        
        super().__init__()

        self.min_length = min_length
        self.mc_files = []
        self.spk2files = {}
        self.speakers = speakers[:]
     
        for spk in self.speakers:
            
            if spk not in self.spk2files:
                self.spk2files[spk] = []
            
            _spk_files = glob.glob(join(data_dir, f'{spk}*.npy'))
            
            mc_files = self.rm_too_short_utt(_spk_files, min_length)

            if few_shot is not None:
                assert isinstance(few_shot, int)
                assert few_shot < len(mc_files), f'speaker {spk} training samples less than few shot limit'
                mc_files = mc_files[: few_shot]
                print(f"speaker {spk} few shot training sampels {len(mc_files)}", flush=True)
            
            self.spk2files[spk].extend(mc_files)
            
            _spk_files_tup = [(spk, fi) for fi in mc_files]
            self.mc_files.extend(_spk_files_tup)
        
        self.num_files = len(self.mc_files)
        print("Number of samples: ", self.num_files)
        
        
        
        for _, f in self.mc_files:
            mc = np.load(f)
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError(f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!") 
        
    def rm_too_short_utt(self, mc_files, min_length):
        new_mc_files = []
        for mcfile in mc_files:
            mc = np.load(mcfile)
            if mc.shape[0] > min_length:
                new_mc_files.append(mcfile)
        return new_mc_files

    def sample_seg(self, feat, sample_len=None):
        if sample_len is None:
            sample_len = self.min_length
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s+sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        src_spk, src_filename = self.mc_files[index]
        src_mc = np.load(src_filename)
        src_mc = self.sample_seg(src_mc)
        src_mc = np.transpose(src_mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape
        
        src_spk_id = self.speakers.index(src_spk)
        src_spk_cat = np.squeeze(to_categorical([src_spk_id], num_classes=len(self.speakers)))

        # choose another spk
        speakers =self.speakers[:]
        speakers.remove(src_spk)
        trg_spk_index = np.random.randint(0, len(speakers) ) # bug fixed
        
        trg_spk = speakers[trg_spk_index]
        trg_spk_id = self.speakers.index(trg_spk)
        trg_spk_cat = np.squeeze(to_categorical([trg_spk_id], num_classes=len(self.speakers)))

        # choose a trg file
        trg_spk_files = self.spk2files[trg_spk]
        trg_file_index = np.random.randint(0, len(trg_spk_files))
        trg_filename = trg_spk_files[trg_file_index]
        
        # load trg mc and do segmentation
        trg_mc = np.load(trg_filename)
        
        trg_mc = self.sample_seg(trg_mc)
        trg_mc = np.transpose(trg_mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape

        return torch.FloatTensor(src_mc), torch.LongTensor([src_spk_id]).squeeze_(), torch.FloatTensor(src_spk_cat), torch.FloatTensor(trg_mc), torch.LongTensor([trg_spk_id]).squeeze_(), torch.FloatTensor(trg_spk_cat)

class CycDataset(data.Dataset):
    '''dataset for cycle gan training, fix src spk and trg spk'''
    
    def __init__(self, data_dir, src_spk, trg_spk, min_length = 256):
        
        super().__init__()

        self.min_length = min_length
        self.src_spk = src_spk
        self.trg_spk = trg_spk

        src_mc_files = glob.glob(join(data_dir, f'{self.src_spk}_*.npy'))
        trg_mc_files = glob.glob(join(data_dir, f'{self.trg_spk}_*.npy'))

        self.src_mc_files = self.rm_too_short_utt(src_mc_files, min_length)
        self.src_num_files = len(self.src_mc_files)
        print("Number of src samples: ", self.src_num_files)
        
        self.trg_mc_files = self.rm_too_short_utt(trg_mc_files, min_length)
        self.trg_num_files = len(self.trg_mc_files)
        print("Number of trg samples: ", self.trg_num_files)
        
        
        for f in self.src_mc_files:
            mc = np.load(f)
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError(f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!") 
        
        for f in self.trg_mc_files:
            mc = np.load(f)
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError(f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!") 
    def rm_too_short_utt(self, mc_files, min_length):
        new_mc_files = []
        for mcfile in mc_files:
            mc = np.load(mcfile)
            if mc.shape[0] > min_length:
                new_mc_files.append(mcfile)
        return new_mc_files

    def sample_seg(self, feat, sample_len=None):
        if sample_len is None:
            sample_len = self.min_length
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s+sample_len, :]

    def __len__(self):
        return self.src_num_files

    def __getitem__(self, index):
        src_filename = self.src_mc_files[index]
        src_mc = np.load(src_filename)
        src_mc = self.sample_seg(src_mc)
        src_mc = np.transpose(src_mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape
        
        trg_index = np.random.randint(0, self.trg_num_files)
        trg_filename = self.trg_mc_files[trg_index]
        trg_mc = np.load(trg_filename)
        trg_mc = self.sample_seg(trg_mc)
        trg_mc = np.transpose(trg_mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape

        return torch.FloatTensor(src_mc), torch.FloatTensor(trg_mc)


class MyDataset(data.Dataset):
    """Dataset for MCEP features and speaker labels."""
    def __init__(self, data_dir, speakers, min_length = 256, few_shot = None):
        self.min_length = min_length
        self.speakers = speakers[:]
        mc_files = []
        for spk in self.speakers:
            # [0827 new feature]: add few shot learning feature, limit training samples
            if few_shot is not None:
                mc_dirs = list(glob.glob(join(data_dir, f'{spk}_*.npy')))
                mc_dirs = self.rm_too_short_utt(mc_dirs, min_length)
                
                assert isinstance(few_shot, int)
                assert few_shot < len(mc_dirs)
                few_shot_mc_dirs = mc_dirs[:few_shot]
                duration = self.calc_duration(few_shot_mc_dirs, 5)
                print(f"spk {spk} org samps {len(mc_dirs)} few shot samps {len(few_shot_mc_dirs)} duration {duration}", flush=True)
                mc_files.extend(few_shot_mc_dirs)
            else:

                mc_files.extend(glob.glob(join(data_dir, f'{spk}_*.npy')))
                mc_files = self.rm_too_shot_utt(mc_files, min_length)
        
        self.mc_files = mc_files[:]


        #mc_files = glob.glob(join(data_dir, '*.npy'))
        #mc_files = [i for i in mc_files if basename(i)[:4] in speakers] 
        
        #self.mc_files = self.rm_too_short_utt(mc_files, min_length)
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)
        for f in self.mc_files:
            mc = np.load(f)
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError(f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!") 
    
    def calc_duration(self, mc_files, frame_rate):
        
        n_frames = 0
        for mcf in mc_files:
            mc = np.load(mcf)
            frms = mc.shape[0]
            n_frames += frms
        duration = (n_frames * frame_rate) / 1000.0
        return duration
    def rm_too_short_utt(self, mc_files, min_length):
        new_mc_files = []
        for mcfile in mc_files:
            mc = np.load(mcfile)
            if mc.shape[0] > min_length:
                new_mc_files.append(mcfile)
        return new_mc_files

    def sample_seg(self, feat, sample_len=None):
        if sample_len is None:
            sample_len = self.min_length
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s+sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.mc_files[index]
        spk = basename(filename).split('_')[0]
        #spk = basename(dirname(filename))
        if spk not in self.speakers:
            raise Exception(f"speaker {spk} not in self.speakers {self.speakers}")
        spk_idx = self.speakers.index(spk)
        mc = np.load(filename)
        mc = self.sample_seg(mc)
        mc = np.transpose(mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape
        # to one-hot
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(self.speakers)))

        return torch.FloatTensor(mc), torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat)
        
class PairTestDataset(object):
    '''Dataset for testing with pair sample input'''
    
    def __init__(self, data_dir, wav_dir, speakers, src_spk, trg_spk):
        

        self.src_spk = src_spk
        self.trg_spk = trg_spk

        self.mc_files = sorted(glob.glob(join(data_dir, f'{self.src_spk}_*.npy')))
        self.trg_mc_files = sorted(glob.glob(join(data_dir, f'{self.trg_spk}_*.npy')))

        self.src_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(src_spk)))
        self.trg_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(trg_spk)))

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        
        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        
        self.spk_idx = speakers.index(trg_spk)
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat
        
        self.src_spk_idx = speakers.index(src_spk)
        src_spk_cat = to_categorical([self.src_spk_idx], num_classes=len(speakers))
        self.spk_c_src = src_spk_cat

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile)
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            
            trg_index = np.random.randint(0, len(self.trg_mc_files))
            trg_mc_file = self.trg_mc_files[trg_index]
            trg_mc = np.load(trg_mc_file)
            src_mc = np.load(mcfile)
            batch_data.append((wavfile_path, src_mc, trg_mc))
        return batch_data       

class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, data_dir, wav_dir, speakers, src_spk='p262', trg_spk='p272'):
        self.src_spk = src_spk
        self.trg_spk = trg_spk
        self.mc_files = sorted(glob.glob(join(data_dir,f'{self.src_spk}_*.npy')))

        self.src_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(src_spk)))
        self.trg_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(trg_spk)))
        
        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        self.spk_idx = speakers.index(trg_spk)
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat
        org_id = speakers.index(src_spk)
        org_cat = to_categorical([org_id], num_classes = len(speakers))
        self.spk_c_org = org_cat
    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile)
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data       

def get_loader(data_dir, batch_size=32, min_length = 256,mode='train', speakers = None, num_workers=1, few_shot = None):
    dataset = MyDataset(data_dir, speakers, min_length, few_shot = few_shot)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader


if __name__ == '__main__':
    loader = get_loader('./data/mc/train')
    data_iter = iter(loader)
    for i in range(10):
        mc, spk_idx, acc_idx, spk_acc_cat = next(data_iter)
        print('-'*50)
        print(mc.size())
        print(spk_idx.size())
        print(acc_idx.size())
        print(spk_acc_cat.size())
        print(spk_idx.squeeze_())
        print(spk_acc_cat)
        print('-'*50)







