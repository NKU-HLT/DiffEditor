import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset
from utils.spec_aug.time_mask import generate_time_mask, generate_alignment_aware_time_mask, generate_inference_mask


class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]
            
            
        from transformers import BertTokenizer, BertModel
        cache_path="./cache/bert-base-multilingual-cased"

        self.word_tokenizer=BertTokenizer.from_pretrained(cache_path)
        self.word_encoder=BertModel.from_pretrained(cache_path)
        # self.word_proj = nn.Linear(768, self.hidden_size, bias=True)
        
        
        import torch
        import os
        import json
        
        base_dir="./data/processed/vctk"
        from utils.text.text_encoder import is_sil_phoneme, build_token_encoder
        word_encoder = build_token_encoder(f'{base_dir}/word_set.json')
        with open(os.path.join(base_dir,'word_set.json'), 'r') as file:
                    vctk_word = json.load(file)
                    print(len(vctk_word))
                    
        word_proj_vctk=dict()   
    
        for i in vctk_word:
            word_proj_vctk[word_encoder.encode(i)[0]]=self.word_tokenizer.encode(i, add_special_tokens=False) 
            
        word_proj_vctk[word_encoder.encode("<EOS>")[0]]=[102]
        word_proj_vctk[word_encoder.encode("<BOS>")[0]]=[101]
        word_proj_vctk[word_encoder.encode("|")[0]]=None
        self.word_proj_vctk=word_proj_vctk

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item['mel']) == self.sizes[index], (len(item['mel']), self.sizes[index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        ph_token = torch.LongTensor(item['ph_token'][:hparams['max_input_tokens']])
        word_token = torch.LongTensor(item['word_token'])
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
            "ph2word": torch.LongTensor(item["ph2word"]),
            "word_token":word_token,
        }
        if hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
            sample["spk_id"] = int(item['spk_id'])
        if hparams['use_spk_id']:
            sample["spk_id"] = int(item['spk_id'])
        if 'use_reversal_classifier' in hparams and hparams['use_reversal_classifier']:
            sample["spk_id"] = int(item['spk_id'])
            
        # add bert_input and word2bert
                # bert_input=txt_input.clone()
        
        # 把这个word转化为bert的id
        word2bert=torch.LongTensor([])
        bert_input=torch.LongTensor([])
        for idx , word in enumerate(word_token):
                # print(idx)
                # print(word)
                # print(torch.tensor(self.word_proj_vctk[int(word)]))
                if self.word_proj_vctk[int(word)]==None:
                    continue
                elif word==0:
                    break
                else:
                    # print(self.word_proj_vctk[int(word)])
                    n=len(self.word_proj_vctk[int(word)])
                    bert_input= torch.cat((bert_input,torch.tensor(self.word_proj_vctk[int(word)])))   
                    add_tensor = torch.full((n,), idx)
                    word2bert = torch.cat((word2bert,add_tensor))
        # print(bert_input)  
        # print(word2bert)
        sample['bert_input']=bert_input
        sample['word2bert']=word2bert
            
        
        
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = collate_1d_or_2d([s['txt_token'] for s in samples], 0)
        word_tokens = collate_1d_or_2d([s['word_token'] for s in samples], 0)
        bert_token_ids = collate_1d_or_2d([s['bert_input'] for s in samples], 0)
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])
        word2bert  = collate_1d_or_2d([s['word2bert'] for s in samples])
        
        
        
      
        
        # 生成注意力掩码
        attention_mask = (bert_token_ids != 0).long()  # 非填充值的位置为1，填充值位置为0

        # 生成分段IDs（单句子情况，所有分段IDs都为0）
        token_type_ids = torch.zeros_like(bert_token_ids)

        bert_input={'input_ids':bert_token_ids,'attention_mask':attention_mask,'token_type_ids':token_type_ids}

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'word_token': word_tokens,
            'bert_input': bert_input,
            'word2bert':word2bert,
        }
        

        
        
        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        if 'use_reversal_classifier' in hparams and hparams['use_reversal_classifier']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
            
        return batch


class StutterSpeechDataset(BaseSpeechDataset):
    def __getitem__(self, index):
        sample = super(StutterSpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        sample['wav_fn'] = item['wav_fn']
        mel = sample['mel']
        T = mel.shape[0]
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        max_frames = sample['mel'].shape[0]

        ph_token = sample['txt_token']
        if self.hparams['use_pitch_embed']:
            assert 'f0' in item
            pitch = torch.LongTensor(item.get(self.hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if self.hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch

        # Load stutter mask & generate time mask for speech editing
        if 'stutter_mel_mask' in item:
            sample['stutter_mel_mask'] = torch.LongTensor(item['stutter_mel_mask'][:max_frames])
        if self.hparams['infer'] == False:
            mask_ratio = self.hparams['training_mask_ratio']
        else:
            mask_ratio = self.hparams['infer_mask_ratio']
        
        if self.hparams['infer'] == False:
            if self.hparams.get('mask_type') == 'random':
                time_mel_mask = generate_time_mask(torch.zeros_like(sample['mel']), ratio=mask_ratio)
            elif self.hparams.get('mask_type') == 'alignment_aware':
                time_mel_mask = generate_alignment_aware_time_mask(torch.zeros_like(sample['mel']), sample['mel2ph'], ratio=mask_ratio)
                
        else:
            # In inference stage we randomly mask the 50% phoneme spans
            #  50 for vctk 
            # time_mel_mask = generate_inference_mask(torch.zeros_like(sample['mel']), sample['mel2ph'], ratio=mask_ratio)
            while(torch.sum(time_mel_mask))<8:
                print("generated span too short, using phoneme mask,retry")
                time_mel_mask = generate_inference_mask(torch.zeros_like(sample['mel']), sample['mel2ph'], ratio=mask_ratio)
            
            
            
        sample['time_mel_mask'] = time_mel_mask
        
        
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(StutterSpeechDataset, self).collater(samples)
        batch['wav_fn'] = [s['wav_fn'] for s in samples]
        batch['ph2word'] = collate_1d_or_2d([s['ph2word'] for s in samples])
        if self.hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            pitch = collate_1d_or_2d([s['pitch'] for s in samples])
            uv = collate_1d_or_2d([s['uv'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        batch.update({
            'mel2ph': mel2ph,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        })
        if 'stutter_mel_mask' in samples[0]:
            batch['stutter_mel_masks'] = collate_1d_or_2d([s['stutter_mel_mask'] for s in samples], self.hparams.get('stutter_pad_idx', -1))
        batch['time_mel_masks'] = collate_1d_or_2d([s['time_mel_mask'] for s in samples], 0)
        return batch
