import math
import random
from functools import partial
from modules.speech_editing.spec_denoiser.diffusion_utils import *
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torchaudio
from modules.speech_editing.spec_denoiser.fs import FastSpeech
from modules.speech_editing.commons.mel_encoder import MelEncoder
from utils.commons.hparams import hparams
from modules.speech_editing.spec_denoiser.classifier import ReversalClassifier
from torch.nn import Sequential, ModuleList, Linear, ReLU, Dropout, LSTM, Embedding


word_dim=768
word_out_dim=768
class GaussianDiffusion(nn.Module):
    def __init__(self, phone_encoder, out_dims, denoise_fn,
                 timesteps=1000, time_scale=1, loss_type='l1', betas=None, spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.fs = FastSpeech(len(phone_encoder), hparams)
        self.mel_encoder = MelEncoder(hidden_size=self.fs.hidden_size+word_out_dim)
        # self.fs2.decoder = None
        self.mel_bins = out_dims

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = get_noise_schedule_list(
                schedule_mode=hparams['schedule_type'],
                timesteps=timesteps + 1,
                min_beta=0.1,
                max_beta=40,
                s=0.008,
            )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.time_scale = time_scale
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('timesteps', to_torch(self.num_timesteps))      # beta
        self.register_buffer('timescale', to_torch(self.time_scale))      # beta
        self.register_buffer('betas', to_torch(betas))      # beta
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod)) # alphacum_t
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev)) # alphacum_{t-1}

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])
        
        
        if 'language_embedding_dimension' in hparams and hparams['language_embedding_dimension']>0 and "languages" in hparams and len(hparams['languages'])>0:
            self.language_embedding = self._get_embedding(hparams['language_embedding_dimension'], len(hparams['languages']))
            
            
            
        if 'use_reversal_classifier' in hparams and hparams['use_reversal_classifier']:
            self.reversal_classifier = ReversalClassifier(
                hparams['hidden_size'], 
                hparams['hidden_size'], 
                hparams['num_spk'],
                hparams["reversal_gradient_clipping"])
            
            
            
            

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(self, x_start, x_t, t, repeat_noise=False):
        b, *_, device = *x_start.shape, x_start.device
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        noise = noise_like(x_start.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, cond, spk_emb=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x_t.shape, x_t.device
        x_0_pred = self.denoise_fn(x_t, t, cond)

        return self.q_posterior_sample(x_start=x_0_pred, x_t=x_t, t=t)

    @torch.no_grad()
    def interpolate(self, x1, x2, t, cond, spk_emb, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        x = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond, spk_emb)
        x = x[:, 0].transpose(1, 2)
        return self.denorm_spec(x)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def diffuse_trace(self, x_start, mask):
        b, *_, device = *x_start.shape, x_start.device
        trace = [self.norm_spec(x_start).clamp_(-1., 1.) * ~mask.unsqueeze(-1)]
        for t in range(self.num_timesteps):
        # for t in range(1):
            t = torch.full((b,), t, device=device, dtype=torch.long)
            trace.append(
                self.diffuse_fn(x_start, t)[:, 0].transpose(1, 2) * ~mask.unsqueeze(-1)
            )
        return trace

    def diffuse_fn(self, x_start, t, noise=None):
        x_start = self.norm_spec(x_start)
        x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
        zero_idx = t < 0 # for items where t is -1
        t[zero_idx] = 0
        noise = default(noise, lambda: torch.randn_like(x_start))
        out = self.q_sample(x_start=x_start, t=t, noise=noise)
        out[zero_idx] = x_start[zero_idx] # set x_{-1} as the gt mel
        return out



    def _get_embedding(self, embedding_dimension, size=None):
        embedding = Embedding(size, embedding_dimension)
        torch.nn.init.xavier_uniform_(embedding.weight)
        return embedding
    def forward(self, txt_tokens, time_mel_masks, mel2ph, spk_embed,
                ref_mels, f0, uv, language_id,energy=None, 
                infer=False, use_pred_mel2ph=False, use_pred_pitch=False,raw_txt=None,ph2word=None,bert_input=None,word2bert=None,word_token=None):
        b, *_, device = *txt_tokens.shape, txt_tokens.device
        ret = {}
        # print(bert_input.shape)
        ret = self.fs(txt_tokens, time_mel_masks, mel2ph, spk_embed, f0, uv, energy,
                       skip_decoder=True, infer=infer, 
                       use_pred_mel2ph=use_pred_mel2ph, use_pred_pitch=use_pred_pitch,raw_txt=raw_txt,ph2word=ph2word,bert_input=bert_input,word2bert=word2bert,word_token=word_token)
        # print("----------------raw_txt---------------------")
        # print(raw_txt)
        decoder_inp = ret['decoder_inp']
        classifier_inp = ret['classifier_inp']
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        # 加上mask mel embedding
        decoder_inp += self.mel_encoder(ref_mels*(1-time_mel_masks)) * tgt_nonpadding

        classifier_inp += self.mel_encoder(ref_mels*(1-time_mel_masks)) * tgt_nonpadding


        # concat word
        # decoder_inp = torch.cat((decoder_inp,ret['word_emb_expaned']),dim=-1)
        if 'use_reversal_classifier' in hparams and hparams['use_reversal_classifier']:
            speaker_prediction = self.reversal_classifier(classifier_inp)
            ret['speaker_prediction'] = speaker_prediction
            
      


        nonpadding = (mel2ph != 0).float().unsqueeze(1).unsqueeze(1) # [B, T]
        
        # 这里有选择的把language—emb
        # 加上
        if 'language_embedding_dimension' in hparams and hparams['language_embedding_dimension']>0 and "languages" in hparams and len(hparams['languages'])>0:
            # 其中的217是中文说话人的个数
            language = language_id
            # print(language_id)
            
            if infer:
                language = language_id
            else:  
                language = language.unsqueeze(1)
            language = language.expand(time_mel_masks.shape[0], time_mel_masks.shape[1])
            
            # print(language)
            embedded = self.language_embedding(language)
          
            decoder_inp = torch.cat((decoder_inp, embedded), dim=-1) * tgt_nonpadding
        
     
        
        cond = decoder_inp.transpose(1, 2)
        
        



        
        if not infer:
            t = torch.randint(0, self.num_timesteps + 1, (b,), device=device).long()
            # Diffusion
            x_t = self.diffuse_fn(ref_mels, t) * nonpadding
            
            # file_path="/home/chenyang/chenyang_space/speech_editing_and_tts/projects/Speech-Editing-Toolkit/ref_mel.wav"
            # torchaudio.save(file_path,ref_mels[0].cpu(),44100)

            # Predict x_{start}
            if 'gt_context' in  hparams and hparams['gt_context']==True:
                # print("doing gt_context")
                # print(ref_mels.unsqueeze(0).shape)
                # print(x_t.shape)
                # print(time_mel_masks.shape)
                # print(nonpadding.shape)
                x_t=x_t.squeeze(1).transpose(-1,-2)
                x_t=(ref_mels*(1-time_mel_masks)+x_t*time_mel_masks) * tgt_nonpadding
                x_t=x_t.unsqueeze(1).transpose(-1,-2)
                # print(x_t.shape)
                x_0_pred = self.denoise_fn(x_t, t, cond) * nonpadding
            else:
                x_0_pred = self.denoise_fn(x_t, t, cond) * nonpadding

            ret['mel_out'] = x_0_pred[:, 0].transpose(1, 2) # [B, T, 80]
        else:
            t = self.num_timesteps  # reverse总步数
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x = torch.randn(shape, device=device)  # noise
            for i in tqdm(reversed(range(0, t)), desc='ProDiff Teacher sample time step', total=t):
                # b是batchsize，这就是八步逆向
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)  # x(mel), t, condition(phoneme)
                # print(x.shape)
            x = x[:, 0].transpose(1, 2)
            ret['mel_out'] = self.denorm_spec(x)  # 去除norm，实际啥也没干
        return ret

    def norm_spec(self, x):
        return x

    def denorm_spec(self, x):
        return x

    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        return self.fs2.cwt2f0_norm(cwt_spec, mean, std, mel2ph)

    def out2mel(self, x):
        return x