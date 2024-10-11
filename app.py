# coding=gbk
import torch    
import os
import subprocess
from flask import Flask, render_template, request, send_file

app = Flask(__name__, static_folder = "./templates/")

def find_difference(original_text, target_text):
    # 确定较短的文本长度
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
    
    # 找到第一个不同的字符的索引
    start_index = 0
    while start_index < min_length and original_text[start_index] == target_text[start_index]:
        start_index += 1
    
    # 找到最后一个不同的字符的索引
    end_index = min_length - 1
    while end_index >= start_index and shorter_one[end_index] == longer_one[diff_length + end_index]:
        end_index -= 1
    
    # 返回不同区间的起始和结束索引
    if start_index >= end_index:
        temp = start_index
        start_index = end_index
        end_index = temp
    if(len(original_text) <= len(target_text)):
        return {'region': f"[{start_index + 1},{end_index + 1}]", 'edited_region': f"[{start_index + 1},{diff_length + end_index + 1}]"}
    else:
        return {'edited_region': f"[{start_index + 1},{end_index + 1}]", 'region': f"[{start_index + 1},{diff_length + end_index + 1}]"}

def find_difference_en(original_text, target_text):
    # 确定较短的文本长度
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
    
    # 找到第一个不同的字符的索引
    start_index = 0
    while start_index < min_length and original_text[start_index] == target_text[start_index]:
        start_index += 1
    
    # 找到最后一个不同的字符的索引
    end_index = min_length - 1
    while end_index >= start_index and shorter_one[end_index] == longer_one[diff_length + end_index]:
        end_index -= 1
    
    # 返回不同区间的起始和结束索引
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
    # 输出命令的标准输出和标准错误
    if len(result.stdout) != 0:
        print("Output:", result.stdout)
    if len(result.stderr) != 0:
        print("Error:", result.stderr)
    
    
    # 检查是否存在 original_audio.wav 文件，如果存在则删除
    if os.path.exists("inference/audio_backup/original_audio.wav"):
        os.remove("inference/audio_backup/original_audio.wav")
    if os.path.exists("inference/out/original_audio.wav"):
        os.remove("inference/out/original_audio.wav")
    if os.path.exists("inference/out/original_audio_ref.wav"):
        os.remove("inference/out/original_audio_ref.wav")
        print("Deleted original_audio.wav")

    # 检查是否存在 target_audio.wav 文件，如果存在则删除
    if os.path.exists("inference/out/target_audio.wav"):
        os.remove("inference/out/target_audio.wav")
        print("Deleted target_audio.wav")
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    
    # 检查是否存在文件和表单数据
    if 'audio' not in request.files:
        return 'No audio file uploaded', 400
    if 'original_text' not in request.form:
        return 'No original text provided', 400
    if 'target_text' not in request.form:
        return 'No target text provided', 400
    
    # 接收音频文件、原文本和目标文本
    audio_file = request.files['audio']
    original_text = request.form['original_text']
    target_text = request.form['target_text']
    
    
    # 检查文件是否为空
    if audio_file.filename == '':
        return 'No selected audio file', 400

    # 检查文本是否为空字符串
    if not original_text.strip():
        return 'Original text is empty', 400
    if not target_text.strip():
        return 'Target text is empty', 400
    
    
    
    # print(f"audio_file: {audio_file.filename}")
    # 保存方式一：
    audio_file_path = 'inference/audio_backup/original_audio.wav'
    audio_file.save(audio_file_path)
    
    # 保存方式二：
    # import torchaudio
    # sample_rate = 44100
    # torchaudio.save(audio_file_path, torch.tensor(audio_file), sample_rate)
    
    # 保存方式三：
    # audio_data = audio_file.read()
    # import wave
    # # 将数据保存为 WAV 文件
    # with wave.open(audio_file_path, 'wb') as wf:
    #     wf.setnchannels(2)  # 设置单声道
    #     wf.setsampwidth(2)  # 设置样本宽度（以字节为单位）
    #     wf.setframerate(48000)  # 设置采样率为 48000 Hz 这里设置的不准确容易导致变声；
    #     wf.writeframes(audio_data)
    
    # 保存方式四：
    # blobData = request.files['data']
    # from werkzeug.utils import secure_filename
    # filename = secure_filename(audio_file.filename)
    # filepath = os.path.join("inference/audio_backup/", audio_file.filename)
    # audio_file.save(filepath)
    
    # 保存方式五：
    # import soundfile
    # data, samplerate = soundfile.read(audio_file)
    # soundfile.write(audio_file_path, data, samplerate, subtype='PCM_16')

    # return 'Processing done'

    # 进行语音编辑：
    region_info = find_difference_en(original_text, target_text)
    region_info['text'] = original_text
    region_info['edited_text'] = target_text
    # print(region_info)
    
    
    # f"export PYTHONPATH=/home/jiayuhang/jiayuhang_space/code/FluentSpeech/FluentSpeech_ch/Speech-Editing-Toolkit",
    commands = [f"export PYTHONPATH=.",
                f"python inference/tts/spec_denoiser_api.py --text \"{original_text}\" --edited_text \"{target_text}\" --region \"{region_info['region']}\" --edited_region \"{region_info['edited_region']}\" --exp_name DiffEditor --config egs/DiffEditor.yaml"
    ]
    # python ../inference/tts/spec_denoiser_api.py --text "刘嘉玲虽然来不及准备贺礼" --edited_text "梁朝伟虽然来不及准备贺礼" --region "[1,3]" --edited_region "[1,3]" --exp_name spec_denoiser_ai3 --config egs/spec_denoiser_aishell3.yaml
    for i in range(len(commands)):
        print(commands[i])
        result = subprocess.run(commands[i], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # 输出命令的标准输出和标准错误
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
        return "输入的原文本与音频文件不一致！请检查是否存在中文、阿拉伯数字等暂不支持的符号。"
    
    return '处理完成！点击确认后即可查看编辑后的音频。'

@app.route('/download')
def download():
    # 提供目标音频文件下载
    target_audio_path = 'inference/out/target_audio.wav'
    if os.path.exists("inference/out/target_audio.wav"):
        return send_file(target_audio_path, as_attachment=True)
    else:
        print("target_audio not exist.")

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
