import os
import argparse
from cycgan.solver import Solver
from data_loader import CycDataset, TestDataset
from torch.backends import cudnn
from torch.utils import data
import json

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.speaker_path):
        raise Exception(f"speaker list {config.speaker_path} does not exist")
    
    with open(config.speaker_path) as f:
        speakers = json.load(f)
    print(f"load speakers {speakers}", flush=True)
    
    # Data loader.
    #train_loader = get_loader(config.train_data_dir, config.batch_size, config.min_length, 'train', speakers, num_workers=config.num_workers,)
    #test_loader = TestDataset(config.test_data_dir, config.wav_dir, speakers, src_spk=config.test_src_spk, trg_spk=config.test_trg_spk)

    dataset = CycDataset(config.train_data_dir, src_spk = config.src_spk, trg_spk = config.trg_spk, min_length = config.min_length)
    train_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=(config.mode=='train'),
                                  num_workers=config.num_workers,
                                  drop_last=True)
    test_loader = TestDataset(config.test_data_dir, config.wav_dir, speakers, src_spk=config.test_src_spk, trg_spk=config.test_trg_spk)
    # Solver for training and testing StarGAN.
    solver = Solver(train_loader, test_loader, config)

    if config.mode == 'train':    
        solver.train()

    # elif config.mode == 'test':
    #     solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--num_speakers', type=int, default=10, help='dimension of speaker labels')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--min_length', type=int, default=256 )
    parser.add_argument('--num_iters', type=int, default=500000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=200000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--src_spk', type = str, default = 'VCC2SF1')
    parser.add_argument('--trg_spk', type = str, default = 'VCC2SM1')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')
    parser.add_argument('--test_src_spk', type = str, default = 'VCC2SF1')
    parser.add_argument('--test_trg_spk', type = str, default = 'VCC2SM1')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--train_data_dir', type=str, default='./data/mc/train')
    parser.add_argument('--test_data_dir', type=str, default='./data/mc/test')
    parser.add_argument('--wav_dir', type=str, default="./data/VCTK-Corpus/wav16")
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--speaker_path', type = str)
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--lambda_id', type=float, default=5, help='weight for id mapping loss')
    
    config = parser.parse_args()
    print(config)
    main(config)
