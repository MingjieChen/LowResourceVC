from cycgan.model import Generator
from cycgan.model import Discriminator
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
from utils import *
from tqdm import tqdm


class Solver(object):
    

    def __init__(self, train_loader, test_loader, config):
        
        # data loader

        self.train_loader = train_loader
        self.test_loader = test_loader

        # model configs

        self.lambda_rec = config.lambda_rec
        self.lambda_id = config.lambda_id
        
        # training configs
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.sampling_rate = config.sampling_rate
        # Test configs

        self.test_iters = config.test_iters
        
        #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        
        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        
        # Build the model and tensorboard.
        self.build_model()
        self.build_tensorboard()
    
    
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)
    
    
    def build_model(self):
        """Create a generator and a discriminator."""
        self.G_src2trg = Generator()
        self.D_trg = Discriminator()
        
        self.G_trg2src = Generator()
        self.D_src = Discriminator()

        self.g_optimizer = torch.optim.Adam(list(self.G_src2trg.parameters()) + list(self.G_trg2src.parameters() ) , self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(list(self.D_src.parameters()) + list(self.D_trg.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G_src2trg, 'G_src2trg')
        self.print_network(self.G_trg2src, 'G_trg2src')
        self.print_network(self.D_src, 'D_src')
        self.print_network(self.D_trg, 'D_trg')
            
        self.G_src2trg.to(self.device)
        self.G_trg2src.to(self.device)
        self.D_src.to(self.device)
        self.D_trg.to(self.device)
    
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        Gs2t_path = os.path.join(self.model_save_dir, '{}-G_s2t.ckpt'.format(resume_iters))
        Gt2s_path = os.path.join(self.model_save_dir, '{}-G_t2s.ckpt'.format(resume_iters))
        Ds_path = os.path.join(self.model_save_dir, '{}-Ds.ckpt'.format(resume_iters))
        Dt_path = os.path.join(self.model_save_dir, '{}-Dt.ckpt'.format(resume_iters))
        self.G_src2trg.load_state_dict(torch.load(Gs2t_path, map_location=lambda storage, loc: storage))
        self.D_src.load_state_dict(torch.load(Ds_path, map_location=lambda storage, loc: storage))
        self.G_trg2src.load_state_dict(torch.load(Gt2s_path, map_location=lambda storage, loc: storage))
        self.D_trg.load_state_dict(torch.load(Dt_path, map_location=lambda storage, loc: storage))
    
    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
    
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
    
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
    def load_wav(self, wavfile, sr=16000):
        wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        return wav_padding(wav, sr=16000, frame_period=5, multiple = 4)  # TODO

    def train(self):
        
        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        test_wavs = [self.load_wav(wavfile, sr = self.sampling_rate) for wavfile in test_wavfiles]
        
        data_iter = iter(self.train_loader)
        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        
        start_iters = 0
        cpsyn_flag = [True, False][0]
        
        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)
        
        
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            
            
            # Fetch data.
            try:
                mc_src, mc_trg  = next(data_iter)
            except:
                data_iter = iter(self.train_loader)
                mc_src, mc_trg = next(data_iter)


            mc_src, mc_trg = mc_src.to(self.device), mc_trg.to(self.device)


            # train discriminator
            
            # src 2 trg discriminator
            d_out_real_trg = self.D_trg(mc_trg)
            d_t_loss_real = nn.MSELoss()(d_out_real_trg, torch.ones_like(d_out_real_trg).to(self.device))
            
            mc_src2trg = self.G_src2trg(mc_src)
            d_out_fake_trg = self.D_trg(mc_src2trg)
            d_t_loss_fake = nn.MSELoss()(d_out_fake_trg, torch.zeros_like(d_out_fake_trg).to(self.device))

            d_trg_loss =  0.5 * (d_t_loss_real + d_t_loss_fake)
            
            # trg 2 src discriminator

            d_out_real_src = self.D_src(mc_src)
            d_s_loss_real = nn.MSELoss()(d_out_real_src, torch.ones_like(d_out_real_src).to(self.device))

            mc_trg2src = self.G_trg2src(mc_trg)
            d_out_fake_src = self.D_src(mc_trg2src)
            d_s_loss_fake = nn.MSELoss()(d_out_fake_src, torch.zeros_like(d_out_fake_src).to(self.device))
            
            d_src_loss = 0.5 * ( d_s_loss_real +  d_s_loss_fake)
            
            d_loss = 4.0 * ( d_trg_loss + d_src_loss )
            
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            # Logging.
            loss = {}
            loss['D/loss_t_real'] = d_t_loss_real.item()
            loss['D/loss_t_fake'] = d_t_loss_fake.item()
            loss['D/loss_s_fake'] = d_s_loss_fake.item()
            loss['D/loss_s_real'] = d_s_loss_real.item()
            #loss['D/loss'] = d_loss.item()

              
            # train generator
            

            #if (i+1) % self.n_critic == 0:
            
            # src 2 trg

            src2trg = self.G_src2trg(mc_src)
            d_out_s2t_fake = self.D_trg(src2trg)
            g_out_s2t_loss = nn.MSELoss()(d_out_s2t_fake, torch.ones_like(d_out_s2t_fake).to(self.device))

            src2trg_back = self.G_trg2src(src2trg)
            src_rec_loss = nn.L1Loss()(src2trg_back, mc_src)

            # trg 2 src

            trg2src = self.G_trg2src(mc_trg)
            d_out_t2s_fake = self.D_src(trg2src)
            g_out_t2s_loss = nn.MSELoss()(d_out_t2s_fake, torch.ones_like(d_out_t2s_fake).to(self.device))
            

            trg2src_back = self.G_src2trg(trg2src)
            trg_rec_loss = nn.L1Loss()(trg2src_back, mc_trg)
            
            # identity
            mc_trg_fake = self.G_src2trg(mc_trg)
            mc_src_fake = self.G_trg2src(mc_src)
            
            loss_rec = src_rec_loss + trg_rec_loss
            loss_id = nn.L1Loss()(mc_trg_fake, mc_trg) + nn.L1Loss()(mc_src_fake, mc_src)
            
            if i > 10000:
                self.lambda_id = 0.

            g_loss = 4.0 * (g_out_s2t_loss + g_out_t2s_loss) + self.lambda_rec * loss_rec + self.lambda_id * loss_id

            self.reset_grad()
            
            g_loss.backward()
            
            self.g_optimizer.step()

            loss['G/loss_s2t_fake'] = g_out_s2t_loss.item()
            loss['G/loss_rec'] = loss_rec.item()
            loss['G/loss_t2s_fake'] = g_out_t2s_loss.item()
            loss['G/loss_id'] = loss_id.item()
            #loss['G/g_loss'] = g_loss.item()   
            
            
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "t [{}],i [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ",{}:{:.6f}".format(tag, value)
                print(log)

                for tag, value in loss.items():
                    self.logger.scalar_summary(tag, value, i+1)
            
            # test
            if (i+1) % self.sample_step == 0:
                sampling_rate=self.sampling_rate
                num_mcep=36
                frame_period=5
                with torch.no_grad():
                    for idx, wav in tqdm(enumerate(test_wavs)):
                        wav_name = basename(test_wavfiles[idx])
                        # print(wav_name)
                        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        f0_converted = pitch_conversion(f0=f0, 
                            mean_log_src=self.test_loader.logf0s_mean_src, std_log_src=self.test_loader.logf0s_std_src, 
                            mean_log_target=self.test_loader.logf0s_mean_trg, std_log_target=self.test_loader.logf0s_std_trg)
                        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                        
                        coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
                        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).to(self.device)
                        #conds = torch.FloatTensor(self.test_loader.spk_c_trg).to(self.device)
                        # print(conds.size())
                        #coded_sp_converted_norm = self.G_src2trg(coded_sp_norm_tensor, conds).data.cpu().numpy()
                        coded_sp_converted_norm = self.G_src2trg(coded_sp_norm_tensor).data.cpu().numpy()
                        coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                        wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted, 
                                                                ap=ap, fs=sampling_rate, frame_period=frame_period)
                        
                        librosa.output.write_wav(
                            join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'), wav_transformed, sampling_rate)
                        if cpsyn_flag:
                            wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp, 
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
                            librosa.output.write_wav(join(self.sample_dir, 'cpsyn-'+wav_name), wav_cpsyn, sampling_rate)
                    cpsyn_flag = False

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                Gs2t_path = os.path.join(self.model_save_dir, '{}-G_s2t.ckpt'.format(i+1))
                Gt2s_path = os.path.join(self.model_save_dir, '{}-G_t2s.ckpt'.format(i+1))
                Ds_path = os.path.join(self.model_save_dir, '{}-Ds.ckpt'.format(i+1))
                Dt_path = os.path.join(self.model_save_dir, '{}-Dt.ckpt'.format(i+1))
                torch.save(self.G_src2trg.state_dict(), Gs2t_path)
                torch.save(self.G_trg2src.state_dict(), Gt2s_path)
                torch.save(self.D_src.state_dict(), Ds_path)
                torch.save(self.D_trg.state_dict(), Dt_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
