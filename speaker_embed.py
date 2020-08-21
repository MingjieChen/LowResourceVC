'''
    This is a script for plot speaker embedding learnt from StarGAN models.
    The speaker embedding is the output vector of the Speaker Enecoder module.
    PCA and T-SNE will be used for plot

'''
import argparse
import torch
from stgan_adain.model import SPEncoder
from stgan_adain.model import SPEncoderPool
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import os
from glob import glob
import numpy as np
import json
def build_speaker_encoder(config):
    
    model = eval(config.spenc_model)(config.num_speakers)
    print(f'Loading the trained Speaker Encoder model from dir {config.model_save_dir} step {config.resume_iters}...', flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)

    model_dir = os.path.join(config.model_save_dir, f'{config.resume_iters}-sp.ckpt')

    model.load_state_dict(torch.load(model_dir, map_location = lambda storage, loc: storage))
    
    model.eval()

    return model, device



def load_input_mc(config, speakers):
    
    speaker2mc = {spk : [] for spk in speakers}
    
    for spk in speakers:
        spk_mc_dirs = list(glob(os.path.join(config.mc_test_dir, spk +'_*.npy')))
        speaker2mc[spk].extend(spk_mc_dirs)
        print(f"spk {spk} load in {len(spk_mc_dirs)} mc files",flush = True)
    return speaker2mc

def _speaker_embeds(model, device, spk_idx, mc_dirs):
    
    ebds_list = []
    spk_label_tensor = torch.LongTensor([spk_idx]).squeeze_().to(device)
    for mc_dir in mc_dirs:
        mc_np = np.load(mc_dir).T # 36, T
        #print(f"load mc shape {mc_np.shape}",flush=True)
        mc_tensor = torch.FloatTensor(np.array(mc_np)).unsqueeze(0).unsqueeze(1)
    
        mc_tensor.to(device)

        embd_tensor = model(mc_tensor, spk_label_tensor)
        embd_np = embd_tensor.squeeze(0).data.cpu().numpy()
        #print(f"embed  {embd_np.shape}",flush=True)
        
        ebds_list.append(embd_np)
    embds_np = np.array(ebds_list)
    print(f"embds_np shape {embds_np.shape}",flush=True)
    return embds_np
        
        

def generate_speaker_embeds(model, device, spk2id, spk2mc_dirs):
    
    spk2embds = {}
    
    for spk in spk2mc_dirs.keys():
        
        spk_idx = spk2id[spk]
        print(f"process speaker {spk} idx {spk_idx}",flush=True)


        mc_dirs = spk2mc_dirs[spk][:]
        embds = _speaker_embeds(model, device, spk_idx, mc_dirs)
        spk2embds[spk] = embds
    return spk2embds


def plot_embedding(config, X, y, idx):
    
    # embedding normalization
    x_min, x_max = np.min(X,0), np.max(X,0)

    X = (X - x_min) / (x_max - x_min)
    
    plt.figure()

    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        plt.text(X[i,0],X[i,1], y[i],
                color = plt.cm.Set1(idx[i] / 10.0), fontdict = {'weight':'bold','size':7}
                )
    plt.savefig(os.path.join(config.plot_output_dir, f'{config.resume_iters}-speaker_embedding.png'))    
    


def run(config):
    
    #load speakers
    with open(config.speaker_path) as f:
        speakers = json.load(f)
    spk2id = {spk:idx for idx, spk in enumerate(speakers)}
    
    os.makedirs(config.plot_output_dir, exist_ok = True)
    os.makedirs(config.save_output_dir, exist_ok = True)

    model, device = build_speaker_encoder(config)
    
    spk2mc_dirs = load_input_mc(config, speakers)
    
    spk2embds = generate_speaker_embeds(model, device, spk2id, spk2mc_dirs)
    
    # save speaker embedding mean vectors
    if config.save:
        for spk in spk2embds.keys():
            emb = spk2embds[spk]
            emb_mean = np.mean(emb, axis = 0)
            os.makedirs(os.path.join(config.save_output_dir, f'{config.resume_iters}'), exist_ok = True)
            np.save(os.path.join(config.save_output_dir,f'{config.resume_iters}', f'{spk}-emd_mean.npy'), emb_mean)
    

    if config.plot:
        # plot 
        all_embds = []
        all_labels = []
        all_idx = []
        for spk in speakers:
            spk_embds = spk2embds[spk]
            all_embds.append(spk_embds)
            all_labels.extend([spk] * spk_embds.shape[0])
            all_idx.extend([spk2id[spk]] * spk_embds.shape[0])
        X = np.concatenate(all_embds, axis = 0)
        print(f"X shape {X.shape}, y {len(all_labels)}",flush=True)
        
        tsne = manifold.TSNE(n_components = 2, init = 'pca', random_state = 123) 
        
        X_tsne = tsne.fit_transform(X)
        
        plot_embedding(config, X, all_labels, all_idx)
        

    

       



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_speakers', type=int, default=10, help='dimension of speaker labels')

    parser.add_argument('--resume_iters', type=int, default=None, help='step to resume for testing.')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--save_output_dir', type=str, default='./speaker_embeds/')
    parser.add_argument('--plot_output_dir', type=str, default='./speaker_embeds/')
    parser.add_argument('--mc_test_dir', type = str, help = 'test mc dir')
    parser.add_argument('--speaker_path', type = str, required = True)
    
    parser.add_argument('--plot', default = False, action = 'store_true')
    parser.add_argument('--save', default = False, action = 'store_true')   
    parser.add_argument('--spenc_model', type = str, default = 'SPEncoder', help = 'speaker encoder model')
    config = parser.parse_args()
    run(config)
