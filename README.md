
<h1 align="center">
<p> DiffEditor: Enhancing Speech Editing with Semantic Enrichment and Acoustic Consistency </p>
</h1>



## Abstract  

As text-based speech editing becomes increasingly prevalent, the demand for unrestricted free-text editing continues to grow. However, existing speech editing techniques encounter significant challenges, particularly in maintaining intelligibility and acoustic consistency when dealing with out-of-domain (OOD) text. In this paper, we introduce DiffEditor, a novel speech editing model designed to enhance performance in OOD text scenarios through semantic enrichment and acoustic consistency. To improve the intelligibility of the edited speech, we enrich the semantic information of phoneme embeddings by integrating word embeddings extracted from a pretrained language model. Furthermore, we emphasize that inter-frame smoothing properties are critical for modeling acoustic consistency, and thus we propose a first-order loss function to promote smoother transitions at editing boundaries and enhance the overall fluency of the edited speech. Experimental results demonstrate that our model achieves state-of-the-art performance in both in-domain and OOD text scenarios. 

### Workflow

![](./resources/workflow.png)


### Implementation
This repo contains official PyTorch implementations of:

- [DiffEditor: Enhancing Speech Editing with Semantic Enrichment and Acoustic Consistency](https://arxiv.org/abs/2409.12992)
[Demo page](https://nku-hlt.github.io/DiffEditor/) | [Code](https://github.com/NKU-HLT/DiffEditor) 


This repo contains unofficial PyTorch implementations of:

- [FluentSpeech: Stutter-Oriented Automatic Speech Editing with Context-Aware Diffusion Models](https://github.com/Zain-Jiang/Speech-Editing-Toolkit) (ACL 2023) 
  [Demo page](https://speechai-demo.github.io/FluentSpeech/)

- [CampNet: Context-Aware Mask Prediction for End-to-End Text-Based Speech Editing](https://arxiv.org/pdf/2202.09950) (ICASSP 2022)  
[Demo page](https://hairuo55.github.io/CampNet)
- [A3T: Alignment-Aware Acoustic and Text Pretraining for Speech Synthesis and Editing](https://proceedings.mlr.press/v162/bai22d/bai22d.pdf) (ICML 2022)  
[Demo page](https://educated-toothpaste-462.notion.site/Demo-b0edd300e6004c508744c6259369a468) | [Official code](https://github.com/richardbaihe/a3t)
- [EditSpeech: A text based speech editing system using partial inference and bidirectional fusion](https://arxiv.org/pdf/2107.01554) (ASRU 2021)  
[Demo page](https://daxintan-cuhk.github.io/EditSpeech/)



## Supported Datasets
Our framework supports the following datasets:

- VCTK

#### Downloading VCTK

You can download the VCTK dataset from the official website. Follow these steps:

1. Visit the [VCTK dataset download page](https://datashare.ed.ac.uk/handle/10283/2651).
4. Download the dataset (VCTK-Corpus.tar.gz).

Extract the downloaded file to your desired directory. For example:

```bash
mkdir data/raw
tar -xzf VCTK-Corpus.tar.gz -C ../data/raw
```

## Install Dependencies
Please install the latest numpy, torch and tensorboard first. Then run the following commands:
```bash
export PYTHONPATH=.
# install requirements.
pip install -U pip
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
```
Finally, install Montreal Forced Aligner following the link below:

`https://montreal-forced-aligner.readthedocs.io/en/latest/`

## Download the pre-trained vocoder
```
mkdir pretrained
mkdir pretrained/hifigan_hifitts
```
download `model_ckpt_steps_2168000.ckpt`, `config.yaml`, from https://drive.google.com/drive/folders/1n_0tROauyiAYGUDbmoQ__eqyT_G4RvjN?usp=sharing to `pretrained/hifigan_hifitts`


## Download the pre-trained bert
```
mkdir cache
```

<!-- https://huggingface.co/google-bert/bert-base-multilingual-cased -->
download `bert-base-multilingual-cased` directory, from https://huggingface.co/google-bert/bert-base-multilingual-cased to `cache/bert-base-multilingual-cased`

## Data Preprocess
```bash
# The default dataset is ``vctk``.
python data_gen/tts/base_preprocess.py --config egs/DiffEditor.yaml
bash data_gen/tts/run_mfa_train_align.sh  vctk  data_gen/tts/mfa_train_config_vctk.yaml
python data_gen/tts/base_binarizer.py --config egs/DiffEditor.yaml
```

## Train
```bash
# Example run for DiffEditor.
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --dir /path/to/your/DiffEditor --config egs/DiffEditor.yaml --exp_name DiffEditor --reset
```


## Inference
We provide the data structure of inference in inference/example.csv. `text` and `edited_text` refer to the original text and target text. `region` refers to the word idx range (start from 1 ) that you want to edit. `edited_region` refers to the word idx range of the edited_text.

|  id   | item_name  | text | edited_text| wav_fn_orig | edited_region| region|
| -- | -- | -- | -- | -- | -- | -- |
|  0  | 1  | "I'd love to be at the world cup." | "I'd <mark>absolutely</mark> love to be at the world cup." | inference/audio_example/1.wav | [1,3] | [1,2] |

```bash
# run with example_en.csv
bash run_test.sh  DiffEditor   inference/example_en.csv  en  inference/raw_wav   1
```
the result is named by DiffEditor_auto in inferecne directory

## Evaluation
Example Objective Evaluation for DiffEditor.
You can use the following objective evaluation metrics: MCD, STOI, PESQ
```bash

python eval/get_metrics.py
```




## License and Agreement
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.


## Tips
1. If you find the ``mfa_dict.txt``, ``mfa_model.zip``, ``phone_set.json``, or ``word_set.json`` are missing in inference, you need to run the preprocess script in our repo to get them. You can also download all of these files you need for inferencing the pre-trained model from
``https://drive.google.com/drive/folders/1BOFQ0j2j6nsPqfUlG8ot9I-xvNGmwgPK?usp=sharing`` and put them in ``data/processed/vctk``. 
2. Please specify the MFA version as 2.0.0rc3.
3. in order to reproduce the paper result, you need to use the testset shown in data/vctk_test_split.csv 




<!-- ## Citing
To cite this repository:
```bibtex
@article{liu2023fluenteditor,
  title={FluentEditor: Text-based Speech Editing by Considering Acoustic and Prosody Consistency},
  author={Liu, Rui and Xi, Jiatian and Jiang, Ziyue and Li, Haizhou},
  journal={Proc. InterSpeech2024},
  year={2024}
}

``` -->

## Author

E-mail：2120230617@mail.nankai.edu.cn
