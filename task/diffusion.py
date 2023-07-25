import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
MIN_MIDI = 21
MAX_MIDI = 108
HOP_LENGTH = 160
SAMPLE_RATE = 16000

import partitura as pt
from utils import animate_sampling, render_sample, Normalization

# from model.utils import Normalization
def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    x_start: x0 (B, 1, T, F)
    t: timestep information (B,)
    """    
    # sqrt_alphas is mean of the Gaussian N() - for the forward process distribution
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t] # extract the value of \bar{\alpha} at time=t
    # one minus alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    # boardcasting into correct shape
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None].to(x_start.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None].to(x_start.device)

    # scale down the input, and scale up the noise as time increases?
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract_x0(x_t, epsilon, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    x_t: The output from q_sample
    epsilon: The noise predicted from the model
    t: timestep information
    """
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t] # extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None].to(x_t.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None].to(x_t.device)    

    # obtaining x0 based on the inverse of eq.4 of DDPM paper
    return (x_t - sqrt_one_minus_alphas_cumprod_t * epsilon) / sqrt_alphas_cumprod_t


class RollDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type,
                 beta_start,
                ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # define beta schedule
        # beta is variance
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas - reparameterization
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # alpha bar
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def training_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)  
        device = batch.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Train/loss", loss)            
      
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)  
        device = batch.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Val/loss", loss)
        
    def test_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)
        device = batch.device
        
        
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Test/loss", loss)        
        
    def predict_step(self, batch, batch_idx):
        # inference code
        # Unwrapping TensorDataset (list)
        # It is a pure noise
        img = batch[0]
        b = img.shape[0] # extracting batchsize
        device=img.device
        imgs = []
        for i in tqdm(reversed(range(0, self.hparams.timesteps)), desc='sampling loop time step', total=self.hparams.timesteps):
            img = self.p_sample(
                           img,
                           i)
            img_npy = img.cpu().numpy()
            
            if (i+1)%10==0:
                for idx, j in enumerate(img_npy):
                    # j (1, T, F)
                    fig, ax = plt.subplots(1,1)
                    ax.imshow(j[0].T, aspect='auto', origin='lower')
                    self.logger.experiment.add_figure(
                        f"sample_{idx}",
                        fig,
                        global_step=self.hparams.timesteps-i)
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            imgs.append(img_npy)
        torch.save(imgs, 'imgs.pt')
    
    def p_losses(self, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = q_sample(x_start=x_start, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
        predicted_noise = self(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
    
    def p_sample(self, x, t_index):
        # x is Guassian noise
        
        # extracting coefficients at time t
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(x, t_tensor) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise             
 
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.hparams.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]


class PCodecDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type,
                 loss_keys,
                 beta_start,
                 beta_end,                 
                 frame_threshold,
                 training,
                 sampling,
                 debug=False,
                 generation_filter=0.0
                ):
        super().__init__()
        
        self.save_hyperparameters()

        self.input_normalize = Normalization(0, 1, 'imagewise')
        
        # define beta schedule (beta is variance)
        self.betas = linear_beta_schedule(beta_start, beta_end, timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1- alphas_cumprod)
        self.inner_loop = tqdm(range(self.hparams.timesteps), desc='sampling loop time step')
        
        self.reverse_diffusion = getattr(self, sampling.type) # bound method cfdg_ddpm_x0, from task config
        self.alphas = alphas

    def training_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)

        # self.log("Train/amt_loss", losses['amt_loss'])
        
        # calculating total loss based on keys give
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            self.log(f"Train/{k}", losses[k])            
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            self.log(f"Val/{k}", losses[k])           
        
        if batch_idx == 0:
            # show sampling loss during validation
            noise_list = self.sampling(batch, batch_idx)
            pcodec_pred, _ = noise_list[-1] # (B, 1, T, F)        
            pcodec_label = batch['p_codec'].unsqueeze(1).cpu()
            sampled_loss = self.p_losses(pcodec_label, torch.tensor(pcodec_pred), loss_type='l2')
            self.log(f"Val/sampled_loss", sampled_loss) 

            if hasattr(self.hparams, 'condition'): # the condition for classifier free
                if self.hparams.condition == 'trainable_spec':
                    fig, ax = plt.subplots(1,1)
                    im = ax.imshow(self.trainable_parameters.detach().cpu(), aspect='auto', origin='lower', cmap='jet')
                    fig.colorbar(im, orientation='vertical')
                    self.logger.experiment.add_figure(f"Val/trainable_uncon", fig, global_step=self.current_epoch)
                    plt.close()


    def test_step(self, batch, batch_idx, save_animation=False):
        noise_list = self.sampling(batch, batch_idx)
        
        # noise_list: [(pred_t, t), ..., (pred_0, 0)]
        pcodec_pred, _ = noise_list[-1] # (B, 1, T, F)        
        pcodec_label = batch['p_codec'].unsqueeze(1).cpu()
        
        if batch_idx==0:  # plot the comparison between prediction and label  
            for idx, (pred, label) in enumerate(zip(pcodec_pred[:4], pcodec_label[:4])):
                fig, ax = plt.subplots(2,4, figsize=(48, 8))
                ax.flatten()[idx].imshow(pred[0].T, aspect='auto', origin='lower')
                ax.flatten()[idx+4].imshow(label[0].T, aspect='auto', origin='lower')
                self.logger.log_image(key=f"Test/pred", images=[fig])
                

        # save the prediction samples (of one batch) and render. 
        np.save('samples/test_sample.npy', pcodec_pred)

        performed_part = render_sample(pcodec_pred, batch, "samples/sample")

        if save_animation:
            t_list = torch.arange(1, self.hparams.timesteps, 5).flip(0)
            if t_list[-1] != self.hparams.timesteps:
                t_list = torch.cat((t_list, torch.tensor([self.hparams.timesteps])), 0)
            fig, axes = plt.subplots(2,4, figsize=(16, 5))

            ax_flat = axes.flatten()
            caxs = []
            for ax in axes.flatten():
                div = make_axes_locatable(ax)
                caxs.append(div.append_axes('right', '5%', '5%'))

            ani = animation.FuncAnimation(fig,
                                        animate_sampling,
                                        frames=tqdm(t_list, desc='Animating'),
                                        fargs=(fig, ax_flat, caxs, noise_list, self.hparams.timesteps),                                          
                                        interval=500,                                          
                                        blit=False,
                                        repeat_delay=1000)
            ani.save('samples/.gif', dpi=80, writer='imagemagick')
            
        hook()
        sampled_loss = self.p_losses(pcodec_label, torch.tensor(pcodec_pred), loss_type='l2')
        frame_p, frame_r, frame_f1, _ = precision_recall_fscore_support(pcodec_label.flatten(),
                                                                        pcodec_pred.flatten() > self.hparams.frame_threshold,
                                                                        average='macro')
        
        self.log(f"Test/sampled_loss", sampled_loss)
        self.log(f"Test/frame_f1", frame_f1) 

        
    # def test_step(self, batch, batch_idx):
    #     """This is for directly predicting x0"""
        
    #     batch_size = batch["frame"].shape[0]
    #     roll_label = self.normalize(batch["frame"]).unsqueeze(1)
    #     waveform = batch["audio"]
    #     device = roll_label.device
    #     # Algorithm 1 line 3: sample t uniformally for every example in the batch

    #     t_index=199
    #     t_tensor = torch.tensor(t_index).repeat(batch_size).to(device)
        
    #     # Equation 11 in the paper
    #     noise = torch.randn_like(roll_label)
    #     roll_pred, spec = self(noise, waveform, t_tensor)
        
    #     roll_pred = roll_pred.cpu()
    #     roll_label = roll_label.cpu()
        
    #     if batch_idx==0:
    #         self.visualize_figure(spec.transpose(-1,-2).unsqueeze(1),
    #                               'Test/spec',
    #                               batch_idx)                

    #         fig, ax = plt.subplots(2,2)
    #         fig1, ax1 = plt.subplots(2,2)
    #         fig2, ax2 = plt.subplots(2,2)
    #         for idx in range(4):
    #             ax.flatten()[idx].imshow(roll_pred[idx][0].T.detach(), aspect='auto', origin='lower')
    #             self.logger.experiment.add_figure(
    #                 f"Test/pred",
    #                 fig,
    #                 global_step=self.hparams.timesteps-t_index)                

    #             ax1.flatten()[idx].imshow(roll_label[idx][0].T, aspect='auto', origin='lower')
    #             self.logger.experiment.add_figure(
    #                 f"Test/label",
    #                 fig1,
    #                 global_step=0)

    #             ax2.flatten()[idx].imshow((roll_pred[idx][0]>self.hparams.frame_threshold).T, aspect='auto', origin='lower')
    #             self.logger.experiment.add_figure(
    #                 f"Test/pred_roll",
    #                 fig2,
    #                 global_step=0)  
    #             plt.close()            
            
            
    #     frame_p, frame_r, frame_f1, _ = precision_recall_fscore_support(roll_label.flatten(),
    #                                                                     roll_pred.flatten()>self.hparams.frame_threshold,
    #                                                                     average='binary')
        
    #     for roll_pred_i, roll_label_i in zip(roll_pred.numpy(), roll_label.numpy()):
    #         # roll_pred (B, 1, T, F)
    #         p_est, i_est = extract_notes_wo_velocity(roll_pred_i[0],
    #                                                  roll_pred_i[0],
    #                                                  onset_threshold=self.hparams.frame_threshold,
    #                                                  frame_threshold=self.hparams.frame_threshold,
    #                                                  rule='rule1'
    #                                                 )
            
    #         p_ref, i_ref = extract_notes_wo_velocity(roll_label_i[0],
    #                                                  roll_label_i[0],
    #                                                  onset_threshold=self.hparams.frame_threshold,
    #                                                  frame_threshold=self.hparams.frame_threshold,
    #                                                  rule='rule1'
    #                                                 )            
            
    #         scaling = self.hparams.spec_args.hop_length / self.hparams.spec_args.sample_rate
    #         # scaling = HOP_LENGTH / SAMPLE_RATE

    #         # Converting time steps to seconds and midi number to frequency
    #         i_ref = (i_ref * scaling).reshape(-1, 2)
    #         p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    #         i_est = (i_est * scaling).reshape(-1, 2)
    #         p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    #         p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)            
        
    #         self.log("Test/Note_F1", f)         
    #     self.log("Test/Frame_F1", frame_f1)        
        
    def predict_step(self, batch, batch_idx):
        noise = batch[0]
        waveform = batch[1]
        # if self.hparams.inpainting_f or self.hparams.inpainting_t:
        #     roll_label = batch[2]
        
        device = noise.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        
        self.inner_loop.refresh()
        self.inner_loop.reset()
        
        noise_list = []
        noise_list.append((noise, self.hparams.timesteps))

        for t_index in reversed(range(0, self.hparams.timesteps)):
            noise, spec = self.reverse_diffusion(noise, waveform, t_index)
            noise_npy = noise.detach().cpu().numpy()
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            noise_list.append((noise_npy, t_index))                       
            self.inner_loop.update()
            #======== Animation saved ===========      
            
        # noise_list is a list of tuple (pred_t, t), ..., (pred_0, 0)
        roll_pred = noise_list[-1][0] # (B, 1, T, F)        

        if batch_idx==0:
            self.visualize_figure(spec.transpose(-1,-2).unsqueeze(1),
                                  'Test/spec',
                                  batch_idx)
            for noise_npy, t_index in noise_list:
                if (t_index+1)%10==0: 
                    fig, ax = plt.subplots(2,2)
                    for idx, j in enumerate(noise_npy):
                        if idx<4:
                            # j (1, T, F)
                            ax.flatten()[idx].imshow(j[0].T, aspect='auto', origin='lower')
                            self.logger.experiment.add_figure(
                                f"Test/pred",
                                fig,
                                global_step=self.hparams.timesteps-t_index)
                            plt.close()
                        else:
                            break

            fig1, ax1 = plt.subplots(2,2)
            fig2, ax2 = plt.subplots(2,2)
            for idx, roll_pred_i in enumerate(roll_pred):
                
                ax2.flatten()[idx].imshow((roll_pred_i[0]>self.hparams.frame_threshold).T, aspect='auto', origin='lower')
                self.logger.experiment.add_figure(
                    f"Test/pred_roll",
                    fig2,
                    global_step=0)  
                plt.close()            

            torch.save(noise_list, 'noise_list.pt')
            # torch.save(spec, 'spec.pt')
            # torch.save(roll_label, 'roll_label.pt')
            
            #======== Begins animation ===========
            t_list = torch.arange(1, self.hparams.timesteps, 5).flip(0)
            if t_list[-1] != self.hparams.timesteps:
                t_list = torch.cat((t_list, torch.tensor([self.hparams.timesteps])), 0)
            ims = []
            fig, axes = plt.subplots(2,4, figsize=(16, 5))

            title = axes.flatten()[0].set_title(None, fontsize=15)
            ax_flat = axes.flatten()
            caxs = []
            for ax in axes.flatten():
                div = make_axes_locatable(ax)
                caxs.append(div.append_axes('right', '5%', '5%'))

            ani = animation.FuncAnimation(fig,
                                          self.animate_sampling,
                                          frames=tqdm(t_list, desc='Animating'),
                                          fargs=(fig, ax_flat, caxs, noise_list, ),                                          
                                          interval=500,                                          
                                          blit=False,
                                          repeat_delay=1000)
            ani.save('algo2.gif', dpi=80, writer='imagemagick')         
            #======== Animation saved ===========
            
        # export as midi
        for roll_idx, np_frame in enumerate(noise_list[-1][0]):
            # np_frame = (1, T, 88)
            np_frame = np_frame[0]
            p_est, i_est = extract_notes_wo_velocity(np_frame, np_frame)

            scaling = HOP_LENGTH / SAMPLE_RATE
            # Converting time steps to seconds and midi number to frequency
            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

            clean_notes = (i_est[:,1]-i_est[:,0])>self.hparams.generation_filter

            save_midi(os.path.join('./', f'clean_midi_e{batch_idx}_{roll_idx}.mid'),
                      p_est[clean_notes],
                      i_est[clean_notes],
                      [127]*len(p_est))
            save_midi(os.path.join('./', f'raw_midi_{batch_idx}_{roll_idx}.mid'),
                      p_est,
                      i_est,
                      [127]*len(p_est)) 
            
    def visualize_figure(self, tensors, tag, batch_idx):
        fig, ax = plt.subplots(2,2)
        for idx, tensor in enumerate(tensors): # visualize only 4 piano rolls
            # roll_pred (1, T, F)
            ax.flatten()[idx].imshow(tensor[0].T.cpu(), aspect='auto', origin='lower')
        self.logger.experiment.add_figure(f"{tag}", fig, global_step=self.current_epoch)
        plt.close()
        
    def step(self, batch):
        p_codec, s_codec = batch['p_codec'], batch['s_codec']
        batch_size = p_codec.shape[0]
        device = p_codec.device

        p_codec = self.input_normalize(p_codec)
        p_codec = p_codec.unsqueeze(1)  # (B, 1, T, F)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ## sampling the same t within each batch, might not work well
        # t = torch.randint(0, self.hparams.timesteps, (1,), device=device)[0].long() # [0] to remove dimension
        # t_tensor = t.repeat(batch_size).to(roll.device)
        
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long() # more diverse sampling
        
        noise = torch.randn_like(p_codec) # creating label noise
        
        x_t = q_sample( # sampling noise at time t
            x_start=p_codec,
            t=t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=noise)
        
        # train to predict the noise, loss is computed for the noise
        if self.hparams.training.mode == 'epsilon':
            if self.hparams.debug==True: # When debugging model is use, change waveform into roll
                epsilon_pred, spec = self(x_t, roll, t) # predict the noise N(0, 1)
            else:
                epsilon_pred, spec = self(x_t, waveform, t) # predict the noise N(0, 1)
            diffusion_loss = self.p_losses(noise, epsilon_pred, loss_type=self.hparams.loss_type)

            pred_roll = extract_x0(
                x_t,
                epsilon_pred,
                t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)
            
        # train to predict p_codec
        elif self.hparams.training.mode == 'x_0':
            pred_p_codec, _ = self(x_t, s_codec, t) 
            diffusion_loss = self.p_losses(p_codec, pred_p_codec, loss_type=self.hparams.loss_type)
            
        elif self.hparams.training.mode == 'ex_0':
            epsilon_pred, spec = self(x_t, waveform, t) # predict the noise N(0, 1)
            pred_roll = extract_x0(
                x_t,
                epsilon_pred,
                t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)            
            diffusion_loss = self.p_losses(roll, pred_roll, loss_type=self.hparams.loss_type)   
            
        else:
            raise ValueError(f"training mode {self.training.mode} is not supported. Please either use 'x_0' or 'epsilon'.")
                   
        tensors = {
            "pred_pcodec": pred_p_codec,
            "label_pcodec": p_codec
        }
        losses = {
            "diffusion_loss": diffusion_loss,
        }                   
        
        return losses, tensors
    
    def sampling(self, batch, batch_idx):
        p_codec = batch['p_codec'].unsqueeze(1) 
        s_codec = batch['s_codec']

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        
        self.inner_loop.refresh()
        self.inner_loop.reset()
        
        noise = torch.randn_like(p_codec)
        noise_list = []
        noise_list.append((noise, self.hparams.timesteps))

        for t_index in reversed(range(0, self.hparams.timesteps)):
            noise, _ = self.reverse_diffusion(noise, s_codec, t_index) # cfdg_ddpm_x0
            noise_npy = noise.detach().cpu().numpy()
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            noise_list.append((noise_npy, t_index))                       
            self.inner_loop.update()
        
        return noise_list
        
    def p_losses(self, label, prediction, loss_type="l1"):
        if loss_type == 'l1':
            loss = F.l1_loss(label, prediction)
        elif loss_type == 'l2':
            loss = F.mse_loss(label, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(label, prediction)
        else:
            raise NotImplementedError()

        return loss
    
    def ddpm(self, x, waveform, t_index):
        # x is Guassian noise
        
        # extracting coefficients at time t
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean, spec
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return (model_mean + torch.sqrt(posterior_variance_t) * noise), spec
        
    def ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec
    
    def ddim_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)

        if t_index == 0:
            sigma = 0
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = 0                 
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec               
        
    def ddim(self, x, waveform, t_index):
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)
        
        if t_index == 0:
            model_mean = (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index] 
        else:
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * (
                (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index]) + (
                self.sqrt_one_minus_alphas_cumprod[t_index-1] * epsilon)
            
        return model_mean, spec

    def ddim2ddpm(self, x, waveform, t_index):
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)

        if t_index == 0:   
            model_mean = (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * (
                (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index]) + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * epsilon) + sigma * torch.randn_like(x)

        return model_mean, spec           
        
    # def cfdg_ddpm(self, x, waveform, t_index):
    #     # x is Guassian noise
        
    #     # extracting coefficients at time t
    #     betas_t = self.betas[t_index]
    #     sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
    #     sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

    #     # boardcasting t_index into a tensor
    #     t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
    #     # Equation 11 in the paper
    #     # Use our model (noise predictor) to predict the mean 
    #     epsilon_c, spec = self(x, waveform, t_tensor)
    #     epsilon_0, _ = self(x, torch.zeros_like(waveform), t_tensor)
    #     epsilon = (1+self.hparams.sampling.w)*epsilon_c - self.hparams.sampling.w*epsilon_0
        
    #     model_mean = sqrt_recip_alphas_t * (
    #         x - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
    #     )
    #     if t_index == 0:
    #         return model_mean, spec
    #     else:
    #         # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
    #         posterior_variance_t = self.posterior_variance[t_index]
    #         noise = torch.randn_like(x)
    #         # Algorithm 2 line 4:
    #         return (model_mean + torch.sqrt(posterior_variance_t) * noise), spec     
        
        
    def cfdg_ddpm_x0(self, x, pcodec, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Algo 1 line 4: Use model (noise predictor) to predict the mean and weight the conditioned & unconditioned
        x0_pred_c, spec = self(x, pcodec, t_tensor)
        if type(pcodec) != type(None):
            x0_pred_0, _ = self(x, torch.zeros_like(pcodec), t_tensor, sampling=True) # if sampling = True, the input condition will be overwritten
            # wait... is this really weighting???
            x0_pred = (1 + self.hparams.sampling.w) * x0_pred_c - self.hparams.sampling.w * x0_pred_0
        else:
            x0_pred = x0_pred_c

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            # sigma is the noise magnitude (variance to scale noise)
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index - 1] / self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1 - self.alphas[t_index]))           
            model_mean = (self.sqrt_alphas_cumprod[t_index - 1]) * x0_pred + (
                            torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index - 1] ** 2 - sigma ** 2) * (
                            x - self.sqrt_alphas_cumprod[t_index] * x0_pred) / self.sqrt_one_minus_alphas_cumprod[t_index]) + (  # epsilon
                            sigma * torch.randn_like(x))

        return model_mean, spec
    
    def generation_ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor, sampling=True) # if sampling = True, the input condition will be overwritten
        x0_pred = x0_pred_0
        
        # x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, _
    
    def inpainting_ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_c, spec = self(x, waveform, t_tensor, inpainting_t=self.hparams.inpainting_t, inpainting_f=self.hparams.inpainting_f)
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor, sampling=True) # if sampling = True, the input condition will be overwritten
        x0_pred = (1+self.hparams.sampling.w)*x0_pred_c - self.hparams.sampling.w*x0_pred_0
        # x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec
    
    def cfdg_ddim_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_c, spec = self(x, waveform, t_tensor)
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor)
        x0_pred = (1+self.hparams.sampling.w)*x0_pred_c - self.hparams.sampling.w*x0_pred_0
        # x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = 0 
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = 0          
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec            

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = TriStageLRSchedule(optimizer,
        #                                [1e-8, self.hparams.lr, 1e-8],
        #                                [0.2,0.6,0.2],
        #                                max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
        # scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

        # return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]

    def animate_sampling(self, t_idx, fig, ax_flat, caxs, noise_list):
        # noise_list: Tuple of (x_t, t), (x_t-1, t-1), ... (x_0, 0)
        # x_t (B, 1, T, F)
        # clearing figures to prevent slow down in each iteration.d
        fig.canvas.draw()
        for idx in range(len(noise_list[0][0])): # visualize only 4 piano rolls in a batch
            ax_flat[idx].cla()
            ax_flat[4+idx].cla()
            caxs[idx].cla()
            caxs[4+idx].cla()     

            # roll_pred (1, T, F)
            im1 = ax_flat[idx].imshow(noise_list[0][0][idx][0].detach().T.cpu(), aspect='auto', origin='lower')
            im2 = ax_flat[4+idx].imshow(noise_list[1 + self.hparams.timesteps - t_idx][0][idx][0].T, aspect='auto', origin='lower')
            fig.colorbar(im1, cax=caxs[idx])
            fig.colorbar(im2, cax=caxs[4+idx])

        fig.suptitle(f't={t_idx}')
        row1_txt = ax_flat[0].text(-400,45,f'Gaussian N(0,1)')
        row2_txt = ax_flat[4].text(-300,45,'x_{t-1}')       
        
# functions for roll2midi

def postprocess_probabilities_to_midi_events(output_dict):
    # TODO refactor classes_num, post_processor
    r"""Postprocess probabilities to MIDI events using thresholds.
    Args:
        output_dict, dict: e.g., {
            'frame_output': (N, 88*5),
            'reg_onset_output': (N, 88*5),
            ...}
    Returns:
        midi_events: dict, e.g.,
            {'0': [
                ['onset_time': 130.24, 'offset_time': 130.25, 'midi_note': 33, 'velocity': 100],
                ['onset_time': 142.77, 'offset_time': 142.78, 'midi_note': 33, 'velocity': 100],
                ...]
             'percussion': [
                ['onset_time': 6.57, 'offset_time': 6.70, 'midi_note': 36, 'velocity': 100],
                ['onset_time': 8.13, 'offset_time': 8.29, 'midi_note': 36, 'velocity': 100],
                ...],
             ...}
    """
    midi_events = {}
    for k, plugin_name in enumerate(plugin_ids):
        plugin_name = IX_TO_NAME[plugin_name.item()]        
#         print('Processing plugin_name: {}'.format(plugin_name), end='\r')

        if plugin_name == 'percussion':
            (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                plugins_output_dict[plugin_name],
                detect_type='percussion',
            )

        else:
            (est_note_events, est_pedal_events) = post_processor.output_dict_to_midi_events(
                plugins_output_dict[plugin_name],
                detect_type='piano',
            )
        midi_events[plugin_name] = est_note_events
    return midi_events

def extract_notes_wo_velocity_torch(onsets, frames, onset_threshold=0.5, frame_threshold=0.5, rule='rule1'):
    """
    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = (onsets > onset_threshold).long()
    frames = (frames > frame_threshold).long()
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1 # Make sure the activation is only 1 time-step
    
    if rule=='rule2':
        pass
    elif rule=='rule1':
        # Use in simple models
        onset_diff = onset_diff & (frames==1) # New condition such that both onset and frame on to get a note
    else:
        raise NameError('Please enter the correct rule name')

    pitches = []
    intervals = []

    for nonzero in torch.nonzero(onset_diff):
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame

        # This while loop is looking for where does the note ends
        while onsets[offset, pitch] or frames[offset, pitch]:
            offset += 1
            if offset == onsets.shape[0]:
                break

        # After knowing where does the note start and end, we can return the pitch information (and velocity)        
        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return pitches, intervals


def extract_notes_wo_velocity(onsets, frames, onset_threshold=0.5, frame_threshold=0.5, rule='rule1'):
    """
    Finds the note timings based on the onsets and frames information
    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = (onsets > onset_threshold).astype(int)
    frames = (frames > frame_threshold).astype(int)
    onset_diff = np.concatenate([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1 # Make sure the activation is only 1 time-step
    
    if rule=='rule2':
        pass
    elif rule=='rule1':
        # Use in simple models
        onset_diff = onset_diff & (frames==1) # New condition such that both onset and frame on to get a note
    else:
        raise NameError('Please enter the correct rule name')

    pitches = []
    intervals = []
    
    frame_locs, pitch_locs = np.nonzero(onset_diff)
    for frame, pitch in zip(frame_locs, pitch_locs):

        onset = frame
        offset = frame

        # This while loop is looking for where does the note ends
        while onsets[offset, pitch] or frames[offset, pitch]:
            offset += 1
            if offset == onsets.shape[0]:
                break

        # After knowing where does the note start and end, we can return the pitch information (and velocity)        
        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])

    return np.array(pitches), np.array(intervals)

def save_midi(path, pitches, intervals, velocities):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)