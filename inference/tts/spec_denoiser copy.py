import os
import numpy as np
import torch
import sys
import shutil
import pandas as pd
from tqdm import tqdm
from utils.commons.hparams import hparams, set_hparams
BASE_DIR = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from data_gen.tts.base_preprocess import BasePreprocessor
# from inference_acl.tts.base_tts_infer import BaseTTSInfer
# from inference_acl.tts.infer_utils import get_align_from_mfa_output, extract_f0_uv
from inference.tts.base_tts_infer import BaseTTSInfer
from inference.tts.infer_utils import get_align_from_mfa_output, extract_f0_uv
from modules.speech_editing.spec_denoiser.spec_denoiser import GaussianDiffusion
from modules.speech_editing.spec_denoiser.diffnet import DiffNet
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.spec_aug.time_mask import generate_time_mask
from utils.text.text_encoder import is_sil_phoneme
from resemblyzer import VoiceEncoder
from utils.audio import librosa_wav2spec
from inference.tts.infer_utils import get_words_region_from_origintxt_region, parse_region_list_from_str
from torchsummary import summary
from data_gen.tts.txt_processors.zh import TxtProcessor as zhTxtProcessor
from data_gen.tts.txt_processors.en import TxtProcessor as enTxtProcessor

import os
import torch
from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.commons.hparams import hparams
import utils
import numpy as np
from resemblyzer import VoiceEncoder
from data_gen.tts.txt_processors.zh import TxtProcessor as zhTxtProcessor
from data_gen.tts.txt_processors.en import TxtProcessor as enTxtProcessor
from data_gen.tts.base_preprocess import BasePreprocessor
import librosa
import torch, torchaudio
embedding_dim=768
DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class SpecDenoiserInfer(BaseTTSInfer):
    def __init__(self, hparams, device=None):
        if device is None:
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor = BasePreprocessor()
        self.ph_encoder, self.word_encoder = self.preprocessor.load_dict(self.data_dir)
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()  # 这里开始换了hp
        self.vocoder.eval()
        self.vocoder.to(self.device)
        self.spk_embeding = VoiceEncoder(device='cpu')
        # 由于self.vocoder = self.build_vocoder()将hparams设置为了pretrained里面的参数，导致了部分参数的丢失，这里我们继承一下参数
        from transformers import BertTokenizer, BertModel
        cache_path="/home/chenyang/chenyang_space/backup/Speech-Editing-Toolkit_add_word_embedding_vctk/cache/bert-base-multilingual-cased"

        self.word_tokenizer=BertTokenizer.from_pretrained(cache_path)
        self.word_encoder_bert=BertModel.from_pretrained(cache_path)
        # self.word_proj = nn.Linear(768, self.hidden_size, bias=True)
        
        
        import torch
        import os
        import json
        
        base_dir="/home/chenyang/chenyang_space/backup/Speech-Editing-Toolkit_add_word_embedding_vctk/data/processed/vctk"
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
        

    def build_model(self):
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        load_ckpt(model, hparams['work_dir'], 'model')
        model.to(self.device)
        model.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)

        # Forward the edited txt to the encoder
        edited_txt_tokens = sample['edited_txt_tokens']
        mel = sample['mel']
        mel2ph = sample['mel2ph']
        mel2word = sample['mel2word']
        dur = sample['dur']
        ph2word = sample['ph2word']
        edited_ph2word = sample['edited_ph2word']
        f0 = sample['f0']
        uv = sample['uv']
        words_region = sample['words_region']
        edited_words_region = sample['edited_words_region']
        text = sample['text']
        

        edited_word_idx = words_region[0]
        # edited_word_idx是原始region的
        changed_idx = edited_words_region[0]

        ret = {}
        # print(summary(self.model.fs)) 
        
        
        # print(hparams)
        
        if 'use_separate_encoder' in hparams and hparams['use_separate_encoder'] == True:
            encoder_out_zh = self.model.fs.encoder_zh(edited_txt_tokens)
            encoder_out_en = self.model.fs.encoder_en(edited_txt_tokens)
            mask = torch.zeros_like(edited_txt_tokens)
            
            mask = torch.where((edited_txt_tokens>=1) & (edited_txt_tokens<=166),torch.tensor(1),torch.tensor(0))
            mask = mask.unsqueeze(-1)
            
            encoder_out = encoder_out_zh*mask + encoder_out_en*(1-mask)
            
        
        else:
            encoder_out = self.model.fs.encoder(edited_txt_tokens)
        
        
        
        # separate encoder时候的代码
        # encoder_out_zh = self.encoder_zh(txt_tokens)
        # encoder_out_en = self.encoder_en(txt_tokens)
        # mask = torch.zeros_like(txt_tokens)
        
        # mask = torch.where((txt_tokens>=1) & (txt_tokens<=166),torch.tensor(1),torch.tensor(0))
        # mask = mask.unsqueeze(-1)
        
        # encoder_out = encoder_out_zh*mask + encoder_out_en*(1-mask)
        
        
        
        # encoder_out = self.model.fs.encoder(edited_txt_tokens)  # [B, T, C] 已经被修改过的文本的phonelist,先把离散的phone转化到实数域，然后使用一阶的conv进行处理
        src_nonpadding = (edited_txt_tokens > 0).float()[:, :, None]
        style_embed = self.model.fs.forward_style_embed(sample['spk_embed'], None)
        # 先输入wav，使用resemblerai得到256维度spk——emded然后linear（256，192）得到192的style_embed
        
        
        
        
        
        
        # 这一部分的masked_dur的策略很简单，就是在一开始我们给定了region也就是原始音频被修改的部分，那么region以外的部分，其实就是没有被修改的部分，也就是两边的部分，可以直接拿来用
        masked_dur = torch.zeros_like(edited_ph2word).to(self.device)
        # 这个# edited_word_idx是原始region的
        masked_dur[:, :ph2word[ph2word<edited_word_idx[0]].size(0)] = dur[:, :ph2word[ph2word<edited_word_idx[0]].size(0)]
        if ph2word.max() > edited_word_idx[1]:
            masked_dur[:, -ph2word[ph2word>edited_word_idx[1]].size(0):] = dur[:, -ph2word[ph2word>edited_word_idx[1]].size(0):]
        
        
        
        # Forward duration model to get the duration and mel2ph for edited text seq (Note that is_editing is set as False to get edited_mel2ph)
        # inp是input的意思
        
        
        
        
        
        word_token_t=sample['edited_word_tokens'][0]
        word2bert=torch.LongTensor([])
        bert_input=torch.LongTensor([])
        # print(sample['text'])
        # print(word_token_t)
        for idx , word in enumerate(word_token_t):
                # print(idx)
                # print(word)
                # print(torch.tensor(self.word_proj_vctk[int(word)]))
                if word==0:
                    break
                elif word==2:
                    print("unknowned word")
                    print(sample['edited_words'][idx])
                    encoded_input = self.word_tokenizer(sample['edited_words'][idx], return_tensors='pt',add_special_tokens=False)
                    
                    n=encoded_input['input_ids'].shape[1]
                    print(n)
                    bert_input= torch.cat((bert_input,torch.tensor(encoded_input['input_ids'][0])))   
                    add_tensor = torch.full((n,), idx)
                    word2bert = torch.cat((word2bert,add_tensor))
                    
                elif self.word_proj_vctk[int(word)]==None:  #也就是未知的word
                    continue
                else:
                    # print(self.word_proj_vctk[int(word)])
                    n=len(self.word_proj_vctk[int(word)])
                    # print(n)
                    bert_input= torch.cat((bert_input,torch.tensor(self.word_proj_vctk[int(word)])))   
                    add_tensor = torch.full((n,), idx)
                    word2bert = torch.cat((word2bert,add_tensor))
    
    
        
        word_token_t = word_token_t.unsqueeze(0).to(self.device)
        bert_token_ids = bert_input.unsqueeze(0).to(self.device)
        
        word2bert  = word2bert.unsqueeze(0).to(self.device)
                    # 生成注意力掩码
        attention_mask = (bert_token_ids != 0).long().to(self.device) # 非填充值的位置为1，填充值位置为0

        # 生成分段IDs（单句子情况，所有分段IDs都为0）
        token_type_ids = torch.zeros_like(bert_token_ids).to(self.device)
        # print(self.device)
        bert_input={'input_ids':bert_token_ids,'attention_mask':attention_mask,'token_type_ids':token_type_ids}
            
        word_emb=self.word_encoder_bert(**bert_input)
       
        # print(word_emb['last_hidden_state'])
        word_emb = word_emb['last_hidden_state']
        # print('word_emb')
        # print(word_emb[0])
        
        # word2bert = torch.LongTensor(word2bert)
        
        # down_sample 到word level
        word_aggregated = torch.zeros(word_token_t.shape[0], word_token_t.shape[1], embedding_dim)
        expanded_word2bert = word2bert.unsqueeze(-1).expand(-1, -1, embedding_dim)
        word_aggregated.scatter_add_(1, expanded_word2bert, word_emb)
        # print('word_aggregated')
        # print(word_aggregated[0])
    
    # # 然后upsample到phoneme emb上去
        # print(ph2word.shape)
        # print()
        word_emb_expaned=torch.zeros(edited_ph2word.shape[0], edited_ph2word.shape[1], embedding_dim)
        for i in range(ph2word.shape[0]):
            word_emb_expaned[i] = word_aggregated[i][edited_ph2word[i]-1]
        
        
        print(encoder_out.shape)
        print(word_emb_expaned.shape)
        # encoder_out_ori=encoder_out.clone()
        # encoder_out = torch.cat((encoder_out,word_emb_expaned),dim=-1)
        
        
        
        
        
        dur_inp = (encoder_out + style_embed) * src_nonpadding
        masked_mel2ph = mel2ph
        # 把修改的部分的mel设置为0
        masked_mel2ph[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])] = 0
        
        
        time_mel_masks_orig = torch.zeros_like(mel2ph).to(self.device)
        time_mel_masks_orig[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])] = 1.0
        # 给出了被修改之后的phone edited_txt_tokens，给出了原始音频对应到phone的没有被修改部分的值，然后他会生成被修改部分的phone可能有多长的时间
        edited_mel2ph = self.model.fs.forward_dur(dur_inp, time_mel_masks_orig, masked_mel2ph, edited_txt_tokens, ret, masked_dur=masked_dur, use_pred_mel2ph=True)
        
        #根据phone和word的关系得到mel和word的关系
        edited_mel2word = torch.Tensor([edited_ph2word[0].numpy()[p - 1] for p in edited_mel2ph[0]]).to(self.device)[None, :]
        #得到中间修改部分的改变量
        length_edited = edited_mel2word[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].size(0) - mel2word[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])].size(0)
        
        edited_mel2ph_ = torch.zeros((1, mel2ph.size(1)+length_edited)).to(self.device)
        head_idx = mel2word[mel2word<edited_word_idx[0]].size(0)
        tail_idx = mel2word[mel2word<=edited_word_idx[1]].size(0) + length_edited
        
        # 这里由于edited_mel2ph得到的结果是全部预测的，所以我们需要保证没有修改的部分是一致的，然而由于word的增多或者减少，不能直接copy
        edited_mel2ph_[:, :head_idx] = mel2ph[:, :head_idx]
        edited_mel2ph_[:, head_idx:tail_idx] = edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])]
        # 就是这一段代码的原因
        if mel2word.max() > edited_word_idx[1]:
            # 这里最后加上2是不对的，因为如果两个都在最后的话mel2ph[mel2word>edited_word_idx[1]]就是eos的idx，-mel2ph[mel2word>edited_word_idx[1]].min()使得eos的idx变为0，+ edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max()使得eos变成edit文本的最后一个phonene的idx，最后+2是不对的，因为只有多家一个字的时候+2因为中间有个|，但是如果是eos那和最后一个字之间没有|分割
            # 修改一下
            if edited_word_idx[1]==mel2word[0][-1]-1 and changed_idx[1]==edited_mel2word[0][-1]-1:
                # 同时两个region都是末尾的情况
                edited_mel2ph_[:, tail_idx:] = mel2ph[mel2word>edited_word_idx[1]] - mel2ph[mel2word>edited_word_idx[1]].min() + edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max() + 1
            else:
                edited_mel2ph_[:, tail_idx:] = mel2ph[mel2word>edited_word_idx[1]] - mel2ph[mel2word>edited_word_idx[1]].min() + edited_mel2ph[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].max() + 2
        edited_mel2ph = edited_mel2ph_.long()

        # Get masked mel by concating the head and tial of the original mel
        length_edited = edited_mel2word[(edited_mel2word>=changed_idx[0]) & (edited_mel2word<=changed_idx[1])].size(0) - mel2word[(mel2word>=edited_word_idx[0]) & (mel2word<=edited_word_idx[1])].size(0)
        head_idx = mel2word[mel2word<edited_word_idx[0]].size(0)
        tail_idx = mel2word[mel2word<=edited_word_idx[1]].size(0) + length_edited


        # Create masked ref mel
        ref_mels = torch.zeros((1, edited_mel2ph.size(1), mel.size(2))).to(self.device)
        T = min(ref_mels.size(1), mel.size(1))
        ref_mels[:, :head_idx, :] = mel[:, :head_idx, :]
        ref_mels[:, tail_idx:, :] = mel[mel2word>edited_word_idx[1]]

        # Get masked frame-level f0 and uv (pitch info)
        edited_f0 = torch.zeros((1, edited_mel2ph.size(1))).to(self.device)
        edited_uv = torch.zeros((1, edited_mel2ph.size(1))).to(self.device)
        edited_f0[:, :head_idx] = f0[:, :head_idx]
        edited_f0[:, tail_idx:] = f0[mel2word>edited_word_idx[1]]
        edited_uv[:, :head_idx] = uv[:, :head_idx]
        edited_uv[:, tail_idx:] = uv[mel2word>edited_word_idx[1]]

        # Create time mask
        time_mel_masks = torch.zeros((1, edited_mel2ph.size(1), 1)).to(self.device)
        time_mel_masks[:, head_idx:tail_idx] = 1.0
        
        with torch.no_grad():
            
        #    这里的spk_id仅仅是用来判断language的
        #    使得eos和sos是和跟他最近的一个phoneme类别一致
        #    
            
 
            # language_id=torch.zeros(1,edited_mel2ph.size(1))
            # language_id[0] = edited_txt_tokens[0][edited_mel2ph[0]-1]
            # language_id = ((language_id>=170)& (language_id<=238)).int()
            
            language_id=torch.ones(1,edited_mel2ph.size(1),dtype=torch.int)

            # print("bert_input")  
            # print(bert_input.shape)  
            # print("word_token")  
            # print(word_token.shape)  
            # print(word2bert)
            # print(word2bert)

            
            
            # 拿到ph2word edited_ph2word
            # 拿到mel2ph（生成的）edited_mel2ph
            # raw_txt=None,ph2word=None,bert_input=None,word2bert=None,word_token=None
                        # add bert_input and word2bert
                    # bert_input=txt_input.clone()
            


            
            output = self.model(edited_txt_tokens, time_mel_masks=time_mel_masks, mel2ph=edited_mel2ph, spk_embed=sample['spk_embed'], 
                    ref_mels=ref_mels, f0=edited_f0, uv=edited_uv, energy=None, infer=True, use_pred_pitch=True,language_id=language_id,raw_txt=sample["text"],ph2word=edited_ph2word,bert_input=bert_input,word2bert=word2bert,word_token=word_token_t)
            # 这里是把生成的修改部分放到原始的mel里面
            
            
            
          
            # 这里是把生成的修改部分放到原始的mel里面
            
            # mel_out=output['mel_out']
            mel_out = output['mel_out'] * time_mel_masks + ref_mels * (1-time_mel_masks)
            wav_out = self.run_vocoder(mel_out)
            wav_gt = self.run_vocoder(sample['mel'])
            # item_name = sample['item_name'][0]
            # np.save(f'inference_acl/mel2ph/{item_name}',output['mel2ph'].cpu().numpy()[0])

        wav_out = wav_out.cpu().numpy()
        wav_gt = wav_gt.cpu().numpy()
        mel_out = mel_out.cpu().numpy()
        mel_gt = sample['mel'].cpu().numpy()
        masked_mel_out = ref_mels.cpu().numpy()
        masked_mel_gt = (sample['mel'] * time_mel_masks_orig[:, :, None]).cpu().numpy()

        return wav_out[0], wav_gt[0], mel_out[0], mel_gt[0], masked_mel_out[0], masked_mel_gt[0]

    def preprocess_input(self, inp):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        # Get ph for original txt
        preprocessor = self.preprocessor
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')
        # print(inp)
        ph, txt, words, ph2word, ph_gb_word = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw)
        ph_token = self.ph_encoder.encode(ph)
        word_token = self.word_encoder.encode(words)
        # Get ph for edited txt
        edited_text_raw = inp['edited_text']
        edited_ph, _, edited_words, edited_ph2word, _ = preprocessor.txt_to_ph(
            preprocessor.txt_processor, edited_text_raw)
        edited_ph_token = self.ph_encoder.encode(edited_ph)
        edited_word_token = self.word_encoder.encode(edited_words)
        # Get words_region
        words = words.split(' ')
        edited_words = edited_words.split(' ')
        region, edited_region = parse_region_list_from_str(inp['region']), parse_region_list_from_str(
            inp['edited_region'])
        words_region = get_words_region_from_origintxt_region(words, region)
        edited_words_region = get_words_region_from_origintxt_region(edited_words, edited_region)

        # Generate forced alignment
        wav = inp['wav']
        mel = inp['mel']
        mfa_textgrid = inp['mfa_textgrid']
        mel2ph, dur = get_align_from_mfa_output(mfa_textgrid, ph, ph_token, mel)
        mel2word = [ph2word[p - 1] for p in mel2ph]  # [T_mel]

        # Extract frame-level f0 and uv (pitch info)
        f0, uv = extract_f0_uv(wav, mel)
        
        
        

        item = {'item_name': item_name, 'text': txt, 'ph': ph,
                'ph2word': ph2word, 'edited_ph2word': edited_ph2word,
                'ph_token': ph_token, 'edited_ph_token': edited_ph_token,"edited_word_token":edited_word_token,"edited_words":edited_words,
                'words_region': words_region, 'edited_words_region': edited_words_region,
                'mel2ph': mel2ph, 'mel2word': mel2word, 'dur': dur,
                'f0': f0, 'uv': uv,
                'mel': mel, 'wav': wav}
        return item
    
    
    
    
    def build_dataloader(self,dataset, shuffle, max_tokens=10000000, max_sentences=20000,
                        required_batch_size_multiple=-1, endless=False, batch_by_size=True, 
                        sampler=None):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.commons.dataset_utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers


        return torch.utils.data.DataLoader(dataset,
                                        collate_fn=dataset.collater,
                                        batch_sampler=batches,
                                        num_workers=num_workers,
                                        pin_memory=False)


        # for knn
    def fast_cosine_dist(self,source_feats: torch.tensor, matching_pool: torch.tensor, device: str = 'cpu') -> torch.tensor:
        """ Like torch.cdist, but fixed dim=-1 and for cosine distance."""
        source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
        matching_norms = torch.norm(matching_pool, p=2, dim=-1)
        dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
        dotprod /= 2
        # 为什么不直接点积呢
        dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
        return dists

    def get_knn_spk_emb(self,dataset_path,topk,src_wav):

        # voice_encoder = VoiceEncoder().cuda()
        # print(dataset_path)
        train_dataset_en=BaseSpeechDataset(prefix='train', shuffle=True, data_dir=dataset_path)
        dataloader_en = self.build_dataloader(train_dataset_en, shuffle=True,batch_by_size=True)


        
        # x, sr = torchaudio.load(ori_path, normalize=True)
        # print(src_wav)
        # print(src_wav.shape)
        ori_wav=src_wav
        ori_spk_emb=self.spk_embeding.embed_utterance(ori_wav)
        # print(ori_spk_emb)
        # print(ori_spk_emb.shape)
        ori_spk_emb=torch.FloatTensor(ori_spk_emb)
        ori_spk_emb=ori_spk_emb.unsqueeze(0)

        target_spk_emb=[]
        from tqdm import tqdm
        # 如何计算一个给定音频的spk_emb
        for batch_idx ,batch in tqdm(enumerate(train_dataset_en)):
            
            target_spk_emb.append(batch["spk_embed"])
            
        
        target_spk_emb = torch.stack(target_spk_emb)


        device="cpu"
        device = torch.device(device)

        synth_set = target_spk_emb.to(device)


        query_seq = ori_spk_emb.to(device)

        dists = self.fast_cosine_dist(query_seq, synth_set, device=device)

        best = dists.topk(k=topk, largest=False, dim=-1)

        out_feats = synth_set[best.indices].mean(dim=1)
        # print(out_feats.shape)
        return out_feats

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        edited_ph2word = torch.LongTensor(item['edited_ph2word'])[None, :].to(self.device)
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :].to(self.device)
        dur = torch.LongTensor(item['dur'])[None, :].to(self.device)
        mel2word = torch.LongTensor(item['mel2word'])[None, :].to(self.device)
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        edited_txt_tokens = torch.LongTensor(item['edited_ph_token'])[None, :].to(self.device)
        edited_word_tokens = torch.LongTensor(item['edited_word_token'])[None, :].to(self.device)
        edited_words = item['edited_words']
        # spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        # masked prediction related
        mel = torch.FloatTensor(item['mel'])[None, :].to(self.device)
        wav = torch.FloatTensor(item['wav'])[None, :].to(self.device)

        # get spk embed
        # 这里咱们尝试使用一个特定的embedding，这里应该已经使用到hifigan的config了，我们在hifigan的config里面机上这个选项
        # 
        if 'spk_emb_wav_fn' in hparams:
            wav2spec_res=librosa_wav2spec(hparams['spk_emb_wav_fn'])
            
            spk_embed = self.spk_embeding.embed_utterance(wav2spec_res['wav'].astype(float))
        else:
            
            spk_embed = self.spk_embeding.embed_utterance(item['wav'].astype(float))
            
            
        # spk_embed 是resembler——ai调用得到的，输入为wav输出为一个256维的向量
        spk_embed = torch.FloatTensor(spk_embed[None, :]).to(self.device)
        
        if 'knn' in hparams:
            print("using knn edit")
            
            spk_embed = spk_embed*(1-hparams['knn']['knn-edit-weight']) + self.get_knn_spk_emb(hparams['knn']['knn-edit-dataset'], hparams['knn']['topk'],item['wav'].astype(float))*hparams['knn']['knn-edit-weight']
        
        
        
        # get frame-level f0 and uv (pitch info)
        f0 = torch.FloatTensor(item['f0'])[None, :].to(self.device)
        uv = torch.FloatTensor(item['uv'])[None, :].to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'ph2word': ph2word,
            'edited_ph2word': edited_ph2word,
            'mel2ph': mel2ph,
            'mel2word': mel2word,
            'dur': dur,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'edited_txt_tokens': edited_txt_tokens,
            'words_region': item['words_region'],
            'edited_words_region': item['edited_words_region'],
            # 'spk_ids': spk_ids,
            'mel': mel,
            'wav': wav,
            'spk_embed': spk_embed,
            'f0': f0,
            'uv': uv,
            'edited_word_tokens':edited_word_tokens,
            'edited_words':edited_words,
        }
        return batch
    @classmethod
    def example_run(cls, dataset_info,test_out_dir):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp
        from utils.audio.io import save_wav
        from utils.plot.plot import plot_mel

        set_hparams()
        infer_ins = cls(hp) #经过该函数之后hparam变成了 pretrained/hifigan的参数 . 这里是使用cls来创建了一个实例

        def infer_one(data_info):
            
            wav2spec_res = librosa_wav2spec(data_info['wav_fn_orig'], fmin=55, fmax=7600, sample_rate=22050)
            
            inp = {
                'item_name': data_info['item_name'],
                'text': data_info['text'],
                'edited_text': data_info['edited_text'],
                'region': data_info['region'],
                'edited_region': data_info['edited_region'],
                'mfa_textgrid': data_info['mfa_textgrid'],
                'mel': wav2spec_res['mel'],
                'wav': wav2spec_res['wav']
            }
            
            wav_out, wav_gt, mel_out, mel_gt, masked_mel_out, masked_mel_gt = infer_ins.infer_once(inp)
            os.makedirs(f'inference/'+test_out_dir, exist_ok=True)
            save_wav(wav_out, 'inference/'+test_out_dir+f'/{inp["item_name"]}.wav', hp['audio_sample_rate'])
            save_wav(wav_gt, 'inference/'+test_out_dir+f'/{inp["item_name"]}_ref.wav', hp['audio_sample_rate'])
            return 1

        os.makedirs('infer_out', exist_ok=True)
        for item in dataset_info:
            infer_one(item)


def load_dataset_info(file_path):
    dataset_frame = pd.read_csv(file_path)
    dataset_info = []
    for index, row in dataset_frame.iterrows():
        row_info = {}
        row_info['item_name'] = row['item_name']
        row_info['text'] = row['text']
        row_info['edited_text'] = row['edited_text']
        row_info['wav_fn_orig'] = row['wav_fn_orig']
        row_info['edited_region'] = row['edited_region']
        row_info['region'] = row['region']
        dataset_info.append(row_info)
    return dataset_info


# preprocess data with forced alignment
def data_preprocess(file_path, input_directory, dictionary_path, acoustic_model_path, output_directory, align=True):
    assert os.path.exists(file_path) and os.path.exists(input_directory) and os.path.exists(acoustic_model_path), \
        f"{file_path},{input_directory},{dictionary_path},{acoustic_model_path}"
    dataset_info = load_dataset_info(file_path)
    for data_info in dataset_info:
        data_info['mfa_textgrid'] = f'{output_directory}/{data_info["item_name"]}.TextGrid'
    if not align:
        print('align  is false')
        return dataset_info

   
    if hparams["language"]=="zh":
        txt_processor = zhTxtProcessor()
    elif hparams["language"]=="en":
        txt_processor = enTxtProcessor()
    elif hparams["language"]=="zh_en":
        txt_processor = {"zh":zhTxtProcessor(),"en":enTxtProcessor()}
    # txt_processor = TxtProcessor()

    # gen  .lab file
    def gen_forced_alignment_info(data_info):
        *_, ph_gb_word = BasePreprocessor.txt_to_ph(txt_processor, data_info['text'])
        tg_fn = f'{input_directory}/{data_info["item_name"]}.lab'
        ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                     for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
        
        with open(tg_fn, 'w') as f_txt:
            ph_gb_word_nosil_tabs = ph_gb_word_nosil.replace(' ', '\t')
            f_txt.write(ph_gb_word_nosil_tabs)
        with open(dictionary_path, 'r') as f:  # update mfa dict for unseen word
            lines = f.readlines()
            
            
            # 把字典里面没有的加入字典里面
        with open(dictionary_path, 'a+') as f:
            for item in ph_gb_word_nosil.split(" "):
                item = item + '\t' + ' '.join(item.split('_')) + '\n'
                if item not in lines:
                    f.writelines([item])
    for item in dataset_info:
        gen_forced_alignment_info(item)
        # print("111111111111111111111111111111111111111")
        # print(item["item_name"])
        # print(data_info['item_name'])
        item_name, wav_fn_orig = data_info['item_name'], data_info['wav_fn_orig']
        os.system(f'cp -f {wav_fn_orig} inference/audio/{item_name}.wav')

    print("Generating forced alignments with mfa. Please wait for about several minutes.")
    mfa_out = output_directory
    if os.path.exists(mfa_out):
        shutil.rmtree(mfa_out)
    command = ' '.join(
        ['mfa align -j 4 --clean', input_directory, dictionary_path, acoustic_model_path, output_directory])

    print(command)
   
    os.system(command)
  

    return dataset_info


if __name__ == '__main__':
    
    # 这里的config一开始是自己确定的那只，之后会变为vocoder的config，设置的时候要小心
    # 修改数据集的时候要修改vocoder的ds_name设置
    
    # you can use 'align' to choose whether using MFA during preprocessing
    
    # # english version
    # set_hparams()
    # set_hparams会收到exp的输入参数
    # test_file_path = 'inference/example.csv'
    # test_wav_directory = 'inference/audio_backup/audio_en'
    # dictionary_path = 'data/processed/vctk/mfa_dict2.txt'
    # acoustic_model_path = 'data/processed/vctk/mfa_model.zip'
    # output_directory = 'inference/audio/mfa_out'
    
    
    
    # # madrian version
    # set_hparams()
    # test_file_path = 'inference/example.csv'
    # test_wav_directory = 'inference/audio_backup/env'
    # dictionary_path = 'data/processed/aishell3/mfa_dict2.txt'
    # acoustic_model_path = 'data/processed/aishell3/mfa_model.zip'
    # output_directory = 'inference/audio/mfa_out'
    
    
    
    
    
    # # code-switch version zh_cs
    # set_hparams()
    
    
    
    # print(hparams['work_dir'].split('/')[1])
    
    # test_file_path = 'inference/test_zh_cs.csv'
    # test_wav_directory = 'inference/test_set/zh_cs'
    # dictionary_path = 'data/processed/talcs/mfa_dict2.txt'
    # acoustic_model_path = 'data/processed/talcs/mfa_model.zip'
    # output_directory = 'inference/test_set/zh_cs/mfa_out'
    # test_out_dir = hparams['work_dir'].split('/')[1]
    
    # # os.system('rm -r inference/audio')
    # # os.makedirs(f'inference/audio', exist_ok=True)
    # dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
    #                                output_directory, align=False)


    # SpecDenoiserInfer.example_run(dataset_info,test_out_dir)
    
    
    
    
    # # code-switch version en_cs
    # set_hparams()
    
    # test_file_path = 'inference/test_en_cs.csv'
    # test_wav_directory = 'inference/test_set/en_cs'
    # dictionary_path = 'data/processed/talcs/mfa_dict2.txt'
    # acoustic_model_path = 'data/processed/talcs/mfa_model.zip'
    # output_directory = 'inference/test_set/en_cs/mfa_out'
    
    
    # # os.system('rm -r inference/audio')
    # # os.makedirs(f'inference/audio', exist_ok=True)
    # dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
    #                                output_directory, align=False)


    # SpecDenoiserInfer.example_run(dataset_info,test_out_dir)
    
    # 中文为主的以gpt生成的自动edit  zh2zh
    
    
    
    

    
    
    
    

    
    
    
    set_hparams()
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_file_path', type=str, default='')
    parser.add_argument('--test_out_dir', type=str, default='')
    parser.add_argument('--test_wav_directory', type=str, default='')
    
    args, unknown = parser.parse_known_args()


    print(args)
    
    
    test_file_path = args.test_file_path
    test_wav_directory = args.test_wav_directory
    dictionary_path = 'data/processed/talcs/mfa_dict2.txt'
    acoustic_model_path = 'data/processed/talcs/mfa_model.zip'
    output_directory = test_wav_directory+'/mfa_out'

    # output_directory =  '/home/chenyang/chenyang_space/speech_editing_and_tts/projects/Speech-Editing-Toolkit/inference/test_set/zh_cs/mfa_out'
    test_out_dir = args.test_out_dir
    
    # os.system('rm -r inference/audio')
    # os.makedirs(f'inference/audio', exist_ok=True)
    dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
                                   output_directory, align=False)


    SpecDenoiserInfer.example_run(dataset_info,test_out_dir)    
    
    
    
    
    
    
    
    # # 中文为主的以gpt生成的自动edit  zh2zh
    # set_hparams()
    
    # test_file_path = 'test_dir/zh2zh.csv'
    # test_wav_directory = 'test_dir/audio'
    # dictionary_path = 'data/processed/talcs/mfa_dict2.txt'
    # acoustic_model_path = 'data/processed/talcs/mfa_model.zip'
    # output_directory = 'test_dir/audio/mfa_out'
    # test_out_dir = hparams['work_dir'].split('/')[1]+"_auto/zh2zh_knn0.1_zh"
    
    # # os.system('rm -r inference/audio')
    # # os.makedirs(f'inference/audio', exist_ok=True)
    # dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
    #                                output_directory, align=False)


    # SpecDenoiserInfer.example_run(dataset_info,test_out_dir)
    
    
    
    # # zh2en
    # set_hparams()
    
    # test_file_path = 'test_dir/zh2en.csv'
    # test_wav_directory = 'test_dir/audio'
    # dictionary_path = 'data/processed/talcs/mfa_dict2.txt'
    # acoustic_model_path = 'data/processed/talcs/mfa_model.zip'
    # output_directory = 'test_dir/audio/mfa_out'
    # test_out_dir = hparams['work_dir'].split('/')[1]+"_auto/zh2en_knn0.1_zh"
    
    # # os.system('rm -r inference/audio')
    # # os.makedirs(f'inference/audio', exist_ok=True)
    # dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
    #                                output_directory, align=False)


    # SpecDenoiserInfer.example_run(dataset_info,test_out_dir)
    
    
        
    # # zh2cs
    # set_hparams()
    
    # test_file_path = 'test_dir/zh2cs.csv'
    # test_wav_directory = 'test_dir/audio'
    # dictionary_path = 'data/processed/talcs/mfa_dict2.txt'
    # acoustic_model_path = 'data/processed/talcs/mfa_model.zip'
    # output_directory = 'test_dir/audio/mfa_out'
    # test_out_dir = hparams['work_dir'].split('/')[1]+"_auto/zh2cs_zh_id"
    
    # # os.system('rm -r inference/audio')
    # # os.makedirs(f'inference/audio', exist_ok=True)
    # dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
    #                                output_directory, align=False)
 

    # SpecDenoiserInfer.example_run(dataset_info,test_out_dir)
    
    
    
    
    # set_hparams()
    
    # test_file_path = 'inference/trump.csv'
    # test_wav_directory = 'inference/audio_backup'
    # dictionary_path = 'data/processed/talcs/mfa_dict2.txt'
    # acoustic_model_path = 'data/processed/talcs/mfa_model.zip'
    # output_directory = 'inference/audio_backup/mfa_out'
    # test_out_dir = hparams['work_dir'].split('/')[1]+"_auto/trump_knn0.1_zh"
    
    # # os.system('rm -r inference/audio')
    # # os.makedirs(f'inference/audio', exist_ok=True)
    # dataset_info = data_preprocess(test_file_path, test_wav_directory, dictionary_path, acoustic_model_path,
    #                                output_directory, align=True)


    # SpecDenoiserInfer.example_run(dataset_info,test_out_dir) 
    
    

    
    
    