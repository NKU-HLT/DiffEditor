import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import torch.optim
import torch.utils.data

from tasks.speech_editing.dataset_utils import StutterSpeechDataset
from tasks.tts.speech_base import SpeechBaseTask
from utils.audio.align import mel2token_to_dur
from utils.audio.pitch.utils import denorm_f0
from utils.commons.hparams import hparams
from utils.eval.mcd import get_metrics_mels
from tasks.tts.dataset_utils import BaseSpeechDataset
from tasks.tts.tts_utils import parse_mel_losses, parse_dataset_configs, load_data_preprocessor, load_data_binarizer
from tasks.tts.vocoder_infer.base_vocoder import BaseVocoder, get_vocoder_cls
from utils.audio.align import mel2token_to_dur
from utils.audio.io import save_wav
from utils.audio.pitch_extractors import extract_pitch_simple
from utils.commons.base_task import BaseTask
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.dataset_utils import data_loader, BaseConcatDataset
from utils.commons.hparams import hparams
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.commons.tensor_utils import tensors_to_scalars
from utils.metrics.ssim import ssim
from utils.nn.model_utils import print_arch
from utils.nn.schedulers import RSQRTSchedule, NoneSchedule, WarmupSchedule
from utils.nn.seq_utils import weights_nonzero_speech
from utils.plot.plot import spec_to_figure
from utils.text.text_encoder import build_token_encoder
import matplotlib.pyplot as plt
from utils.metrics.ssim import ssim



class SpeechEditingBaseTask(SpeechBaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = StutterSpeechDataset
        self.sil_ph = self.token_encoder.sil_phonemes()

        self.mcd_dict = {'mcd_total': 0, 'num': 0}

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = SpeechEditingBaseTask(dict_size, hparams)

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_embed = sample.get('spk_embed')
        spk_id = sample.get('spk_ids')
        if not infer:
            target = sample['mels']  # [B, T_s, 80]
            mel2ph = sample['mel2ph']  # [B, T_s]
            f0 = sample.get('f0')
            uv = sample.get('uv')
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                f0=f0, uv=uv, infer=False)
            losses = {}
            self.add_mel_loss(output['mel_out'], target, losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            use_gt_f0 = kwargs.get('infer_use_gt_f0', hparams['use_gt_f0'])
            mel2ph, uv, f0 = None, None, None
            if use_gt_dur:
                mel2ph = sample['mel2ph']
            if use_gt_f0:
                f0 = sample['f0']
                uv = sample['uv']
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id,
                                f0=f0, uv=uv, infer=True)
            return output

    
    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2token_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.token_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]
        losses['pdur'] = F.mse_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
        losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
        losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
        # use linear scale for sentence and word duration
        if hparams['lambda_word_dur'] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float() if hparams['pitch_type'] == 'frame' \
            else (sample['txt_tokens'] != 0).float()
        p_pred = output['pitch_pred']
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv'] and hparams['pitch_type'] == 'frame':
            assert p_pred[..., 1].shape == uv.shape, (p_pred.shape, uv.shape)
                                                                                     
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        losses['f0'] = (F.l1_loss(f0_pred, f0, reduction='none') * nonpadding).sum() \
                       / nonpadding.sum() * hparams['lambda_f0']

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams['audio_sample_rate']
        f0_gt = None
        mel_out = model_out['mel_out']
        if sample.get('f0') is not None:
            f0_gt = denorm_f0(sample['f0'][0].cpu(), sample['uv'][0].cpu())
        self.plot_mel(batch_idx, sample['mels'], mel_out, f0s=f0_gt)
        if self.global_step > 0:
            wav_pred = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0_gt)
            self.logger.add_audio(f'wav_val_{batch_idx}', wav_pred, self.global_step, sr)
            # with gt duration
            model_out = self.run_model(sample, infer=True, infer_use_gt_dur=True)
            dur_info = self.get_plot_dur_info(sample, model_out)
            del dur_info['dur_pred']
            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
            self.logger.add_audio(f'wav_gdur_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_gdur_{batch_idx}',
                          dur_info=dur_info, f0s=f0_gt)

        # with pred duration
            if not hparams['use_gt_dur']:
                model_out = self.run_model(sample, infer=True, infer_use_gt_dur=False)
                dur_info = self.get_plot_dur_info(sample, model_out)
                self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_pdur_{batch_idx}',
                              dur_info=dur_info, f0s=f0_gt)
                wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=f0_gt)
                self.logger.add_audio(f'wav_pdur_{batch_idx}', wav_pred, self.global_step, sr)
        # gt wav
        if self.global_step <= hparams['valid_infer_interval']:
            mel_gt = sample['mels'][0].cpu()
            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

    def get_plot_dur_info(self, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2token_to_dur(sample['mel2ph'], T_txt)[0]
        dur_pred = model_out['dur'] if 'dur' in model_out else dur_gt
        txt = self.token_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
        txt = txt.split(" ")
        return {'dur_gt': dur_gt, 'dur_pred': dur_pred, 'txt': txt}

    def test_step(self, sample, batch_idx):
        assert sample['txt_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        outputs = self.run_model(sample, infer=True)
        text = sample['text'][0]
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel_pred = outputs['mel_out'][0].cpu().numpy()
        mel2ph = sample['mel2ph'][0].cpu().numpy()
        time_mel_masks = sample['time_mel_masks'][0].cpu().numpy()
        mel2ph_pred = None
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred])
        mel_pred_seg = mel_pred[time_mel_masks==1]
        wav_pred_seg =self.vocoder.spec2wav(mel_pred_seg)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred_seg, mel_pred_seg, base_fn % 'P_SEG', gen_dir, None, None])
        
        print("--------------process in testing ------------------")
        print("--------------gen dir ------------------")
        print(gen_dir)
        print("--------------work dir ------------------")
        print(hparams['work_dir'])
        
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph])
            mel_gt_seg = mel_gt[time_mel_masks==1]
            wav_gt_seg =self.vocoder.spec2wav(mel_gt_seg)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt_seg, mel_gt_seg, base_fn % 'G_SEG', gen_dir, None, None])
        
        # print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
            'wav_fn_orig': sample['wav_fn'][0],
        }
    def add_diff_loss(self, output, time_mel_masks, target, sample, losses, postfix=''):
        losses[f'diff'] = getattr(self, f'diff_loss')(output, time_mel_masks, target)

    
    
    
    def diff_loss_ssim(self, output, time_mel_masks, target, postfix=''):
        bias=6.0
    
        
        
        
        
        
        output_diff = output[:, 1:] - output[:, :-1]

        target_diff = target[:,1:]- target[:,:-1]
        
        # l1_loss = F.l1_loss(output_diff, target_diff, reduction='none')
        
        weights = weights_nonzero_speech(target_diff-output_diff)
        
        # l1_loss = (l1_loss * weights).sum() / weights.sum()
        
        
        output_diff = output_diff[:, None] + bias
        target_diff = target_diff[:, None] + bias
        ssim_loss = 1 - ssim(output_diff, target_diff, size_average=False)
        
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        
        return ssim_loss
    
    def diff_loss_mse(self, output, time_mel_masks, target, postfix=''):
        

        # 假设你有一个 (seq_len, dim) 形状的 NumPy 数组
       
        # output_diff = output[:, 1:] - output[:, :-1]
        # 计算一阶差分
        output_diff = output[:, 1:] - output[:, :-1]

        target_diff = target[:,1:]- target[:,:-1]
        
        l1_loss = F.mse_loss(output_diff, target_diff, reduction='none')
        
        weights = weights_nonzero_speech(target_diff-output_diff)
        
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        
        return l1_loss
    
    def diff_loss(self, output, time_mel_masks, target, postfix=''):
        

        # 假设你有一个 (seq_len, dim) 形状的 NumPy 数组
       
        # output_diff = output[:, 1:] - output[:, :-1]
        # 计算一阶差分
        output_diff = output[:, 1:] - output[:, :-1]

        target_diff = target[:,1:]- target[:,:-1]
        
        l1_loss = F.l1_loss(output_diff, target_diff, reduction='none')
        
        weights = weights_nonzero_speech(target_diff-output_diff)
        
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        
        return l1_loss*4
    
    def second_diff_loss(self, output, time_mel_masks, target, postfix=''):
        
        # 假设你有一个 (seq_len, dim) 形状的 NumPy 数组
       
        # output_diff = output[:, 1:] - output[:, :-1]
        # 计算一阶差分
        output_diff = output[:, 1:] - output[:, :-1]
        output_diff2 = output_diff[:, 1:] - output_diff[:, :-1]

        target_diff = target[:,1:]- target[:,:-1]
        target_diff2 = target_diff[:,1:]- target_diff[:,:-1]
        
        l1_loss = F.l1_loss(output_diff2, target_diff2, reduction='none')
        
        weights = weights_nonzero_speech(target_diff2-output_diff2)
        
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        
        return l1_loss*4
        
        
    
    def add_second_diff_loss(self, output, time_mel_masks, target, sample, losses, postfix=''):
        losses[f'second_diff'] = getattr(self, f'second_diff_loss')(output, time_mel_masks, target)
        