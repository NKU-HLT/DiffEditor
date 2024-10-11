# coding=gbk
import torch    
import os
import subprocess
from flask import Flask, render_template, request, send_file

app = Flask(__name__, static_folder = "./templates/")

def find_difference(original_text, target_text):
    # ȷ���϶̵��ı�����
    shorter_one = ""
    longer_one = ""
    if(len(original_text) <= len(target_text)):
        shorter_one = original_text
        longer_one = target_text
    else:
        shorter_one = target_text
        longer_one = original_text 
        
    min_length = min(len(original_text), len(target_text))
    max_length = max(len(original_text), len(target_text))
    diff_length = max_length - min_length
    
    # �ҵ���һ����ͬ���ַ�������
    start_index = 0
    while start_index < min_length and original_text[start_index] == target_text[start_index]:
        start_index += 1
    
    # �ҵ����һ����ͬ���ַ�������
    end_index = min_length - 1
    while end_index >= start_index and shorter_one[end_index] == longer_one[diff_length + end_index]:
        end_index -= 1
    
    # ���ز�ͬ�������ʼ�ͽ�������
    if start_index >= end_index:
        temp = start_index
        start_index = end_index
        end_index = temp
    if(len(original_text) <= len(target_text)):
        return {'region': f"[{start_index + 1},{end_index + 1}]", 'edited_region': f"[{start_index + 1},{diff_length + end_index + 1}]"}
    else:
        return {'edited_region': f"[{start_index + 1},{end_index + 1}]", 'region': f"[{start_index + 1},{diff_length + end_index + 1}]"}

def find_difference_en(original_text, target_text):
    # ȷ���϶̵��ı�����
    shorter_one = ""
    longer_one = ""
    original_text= original_text.split()
    # print(original_text)
    target_text= target_text.split()
    # print(target_text)
    if(len(original_text) <= len(target_text)):
        shorter_one = original_text
        longer_one = target_text
    else:
        shorter_one = target_text
        longer_one = original_text 
        
    min_length = min(len(original_text), len(target_text))
    max_length = max(len(original_text), len(target_text))
    diff_length = max_length - min_length
    
    # �ҵ���һ����ͬ���ַ�������
    start_index = 0
    while start_index < min_length and original_text[start_index] == target_text[start_index]:
        start_index += 1
    
    # �ҵ����һ����ͬ���ַ�������
    end_index = min_length - 1
    while end_index >= start_index and shorter_one[end_index] == longer_one[diff_length + end_index]:
        end_index -= 1
    
    # ���ز�ͬ�������ʼ�ͽ�������
    if start_index >= end_index:
        temp = start_index
        start_index = end_index
        end_index = temp
    if(len(original_text) <= len(target_text)):
        return {'region': f"[{start_index + 1},{end_index + 1}]", 'edited_region': f"[{start_index + 1},{diff_length + end_index + 1}]"}
    else:
        return {'edited_region': f"[{start_index + 1},{end_index + 1}]", 'region': f"[{start_index + 1},{diff_length + end_index + 1}]"}


@app.route('/')
def index():
    result = subprocess.run("export PYTHONPATH=.", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # �������ı�׼����ͱ�׼����
    if len(result.stdout) != 0:
        print("Output:", result.stdout)
    if len(result.stderr) != 0:
        print("Error:", result.stderr)
    
    
    # ����Ƿ���� original_audio.wav �ļ������������ɾ��
    if os.path.exists("inference/audio_backup/original_audio.wav"):
        os.remove("inference/audio_backup/original_audio.wav")
    if os.path.exists("inference/out/original_audio.wav"):
        os.remove("inference/out/original_audio.wav")
    if os.path.exists("inference/out/original_audio_ref.wav"):
        os.remove("inference/out/original_audio_ref.wav")
        print("Deleted original_audio.wav")

    # ����Ƿ���� target_audio.wav �ļ������������ɾ��
    if os.path.exists("inference/out/target_audio.wav"):
        os.remove("inference/out/target_audio.wav")
        print("Deleted target_audio.wav")
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    
    # ����Ƿ�����ļ��ͱ�����
    if 'audio' not in request.files:
        return 'No audio file uploaded', 400
    if 'original_text' not in request.form:
        return 'No original text provided', 400
    if 'target_text' not in request.form:
        return 'No target text provided', 400
    
    # ������Ƶ�ļ���ԭ�ı���Ŀ���ı�
    audio_file = request.files['audio']
    original_text = request.form['original_text']
    target_text = request.form['target_text']
    
    
    # ����ļ��Ƿ�Ϊ��
    if audio_file.filename == '':
        return 'No selected audio file', 400

    # ����ı��Ƿ�Ϊ���ַ���
    if not original_text.strip():
        return 'Original text is empty', 400
    if not target_text.strip():
        return 'Target text is empty', 400
    
    
    
    # print(f"audio_file: {audio_file.filename}")
    # ���淽ʽһ��
    audio_file_path = 'inference/audio_backup/original_audio.wav'
    audio_file.save(audio_file_path)
    
    # ���淽ʽ����
    # import torchaudio
    # sample_rate = 44100
    # torchaudio.save(audio_file_path, torch.tensor(audio_file), sample_rate)
    
    # ���淽ʽ����
    # audio_data = audio_file.read()
    # import wave
    # # �����ݱ���Ϊ WAV �ļ�
    # with wave.open(audio_file_path, 'wb') as wf:
    #     wf.setnchannels(2)  # ���õ�����
    #     wf.setsampwidth(2)  # ����������ȣ����ֽ�Ϊ��λ��
    #     wf.setframerate(48000)  # ���ò�����Ϊ 48000 Hz �������õĲ�׼ȷ���׵��±�����
    #     wf.writeframes(audio_data)
    
    # ���淽ʽ�ģ�
    # blobData = request.files['data']
    # from werkzeug.utils import secure_filename
    # filename = secure_filename(audio_file.filename)
    # filepath = os.path.join("inference/audio_backup/", audio_file.filename)
    # audio_file.save(filepath)
    
    # ���淽ʽ�壺
    # import soundfile
    # data, samplerate = soundfile.read(audio_file)
    # soundfile.write(audio_file_path, data, samplerate, subtype='PCM_16')

    # return 'Processing done'

    # ���������༭��
    region_info = find_difference_en(original_text, target_text)
    region_info['text'] = original_text
    region_info['edited_text'] = target_text
    # print(region_info)
    
    
    # f"export PYTHONPATH=/home/jiayuhang/jiayuhang_space/code/FluentSpeech/FluentSpeech_ch/Speech-Editing-Toolkit",
    commands = [f"export PYTHONPATH=.",
                f"python inference/tts/spec_denoiser_api.py --text \"{original_text}\" --edited_text \"{target_text}\" --region \"{region_info['region']}\" --edited_region \"{region_info['edited_region']}\" --exp_name DiffEditor --config egs/DiffEditor.yaml"
    ]
    # python ../inference/tts/spec_denoiser_api.py --text "��������Ȼ������׼������" --edited_text "����ΰ��Ȼ������׼������" --region "[1,3]" --edited_region "[1,3]" --exp_name spec_denoiser_ai3 --config egs/spec_denoiser_aishell3.yaml
    for i in range(len(commands)):
        print(commands[i])
        result = subprocess.run(commands[i], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # �������ı�׼����ͱ�׼����
    if len(result.stdout) != 0:
        print("Output:", result.stdout)
        # print("Successfully Edit!")
    if len(result.stderr) != 0:
        print("Error:", result.stderr)
    # while True:
    if os.path.exists("inference/out/original_audio.wav"):
        import shutil
        shutil.copyfile("inference/out/original_audio.wav", "inference/out/target_audio.wav")
    else:
        return "�����ԭ�ı�����Ƶ�ļ���һ�£������Ƿ�������ġ����������ֵ��ݲ�֧�ֵķ��š�"
    
    return '������ɣ����ȷ�Ϻ󼴿ɲ鿴�༭�����Ƶ��'

@app.route('/download')
def download():
    # �ṩĿ����Ƶ�ļ�����
    target_audio_path = 'inference/out/target_audio.wav'
    if os.path.exists("inference/out/target_audio.wav"):
        return send_file(target_audio_path, as_attachment=True)
    else:
        print("target_audio not exist.")

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
