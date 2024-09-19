import os
import numpy as np
import torch
from data_gen.tts.base_preprocess import BasePreprocessor
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.speech_editing.campnet.campnet import CampNet
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.spec_aug.time_mask import generate_time_mask

class StutterSpeechInfer(BaseTTSInfer):
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
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        ph_dict_size = len(self.ph_encoder)
        word_dict_size = len(self.word_encoder)
        model = CampNet(ph_dict_size, word_dict_size, self.hparams)
        load_ckpt(model, hparams['work_dir'], 'model')
        model.to(self.device)
        model.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        with torch.no_grad():
            output = self.model(
                sample['txt_tokens'],
                sample['word_tokens'],
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                infer=True,
                forward_post_glow=True,
                mels=sample['mel'],
                time_mel_masks=sample['time_mask'],
                # spk_id=sample.get('spk_ids')
            )
            mel_out = output['mel_out_fine'] * sample['time_mask'] + sample['mel'] * (1-sample['time_mask'])
            wav_out = self.run_vocoder(mel_out)
            wav_gt = self.run_vocoder(sample['mel'])

        wav_out = wav_out.cpu().numpy()
        wav_gt = wav_gt.cpu().numpy()
        mel_out = mel_out.cpu().numpy()
        mel_gt = sample['mel'].cpu().numpy()

        return wav_out[0], wav_gt[0], mel_out[0], mel_gt[0]

    def preprocess_input(self, inp):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        preprocessor = self.preprocessor
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')
        ph, txt, word, ph2word, ph_gb_word = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw)
        word_token = self.word_encoder.encode(word)
        ph_token = self.ph_encoder.encode(ph)
        # spk_id = self.spk_map[spk_name]
        
        # masked prediction related
        wav = inp['wav']
        mel = inp['mel']
        time_mask = np.zeros(mel.shape[0])
        time_mask[120:195] = 1.0
        # time_mask = generate_time_mask(torch.Tensor(mel)).numpy()

        item = {'item_name': item_name, 'text': txt, 'ph': ph,
                'ph_token': ph_token, 'word_token': word_token, 'ph2word': ph2word,
                'mel': mel, 'wav': wav, 'time_mask': time_mask}
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        word_tokens = torch.LongTensor(item['word_token'])[None, :].to(self.device)
        word_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        # spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        # masked prediction related
        mel = torch.FloatTensor(item['mel'])[None, :].to(self.device)
        wav = torch.FloatTensor(item['wav'])[None, :].to(self.device)
        time_mask = torch.FloatTensor(item['time_mask'])[None, :, None].to(self.device)
        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'word_tokens': word_tokens,
            'word_lengths': word_lengths,
            'ph2word': ph2word,
            # 'spk_ids': spk_ids,
            'mel': mel, 
            'wav': wav, 
            'time_mask': time_mask
        }
        return batch

    @classmethod
    def example_run(cls):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp
        from utils.audio.io import save_wav
        from utils.audio import librosa_wav2spec
        from inference.tts.infer_utils import plot_mel

        set_hparams()
        wav2spec_res = librosa_wav2spec('inference/audio/p323_290.wav', fmin=55, fmax=7600, sample_rate=22050)
        inp = {
            'text': 'we didnt enjoy the first game , but today they were excellent .',
            'mel': wav2spec_res['mel'],
            'wav': wav2spec_res['wav'],
        }
        infer_ins = cls(hp)
        wav_out, wav_gt, mel_out, mel_gt = infer_ins.infer_once(inp)

        os.makedirs('infer_out', exist_ok=True)
        save_wav(wav_out, f'inference/out/wav_out.wav', hp['audio_sample_rate'])
        save_wav(wav_gt, f'inference/out/wav_gt.wav', hp['audio_sample_rate'])

        plot_mel(mel_out, 'inference/out/mel_out.png')
        plot_mel(mel_gt, 'inference/out/mel_gt.png')


if __name__ == '__main__':
    StutterSpeechInfer.example_run()
