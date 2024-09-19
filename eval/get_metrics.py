from glob import glob
import pandas as pd
from eval.mcd import cal_mcd_with_wave_batch
from eval.stoi import cal_stoi_with_waves_batch
from eval.pesq_metric import cal_pesq_with_waves_batch


if __name__ == '__main__':
    
# aishell3+vctk
    # wavs_dir = 'checkpoints/spec_denoiser_aishell3_vctk_baseline_new/generated_200000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_aishell3_vctk_reversal/generated_200000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_aishell3_new/generated_200000_/wavs/*'
    # wavs_dir = 'checkpoints/spec_denoiser_vctk_new/generated_200000_/wavs/*'wavs_dir = 'checkpoints/spec_denoiser_vctk_new/generated_200000_/wavs/*'
    wavs_dir = 'checkpoints/fluenteditor2/generated_200000_/wavs/*'
    print(wavs_dir)
    mcd_values = cal_mcd_with_wave_batch(wavs_dir)
    stoi_values = cal_stoi_with_waves_batch(wavs_dir)
    pesq_values = cal_pesq_with_waves_batch(wavs_dir)
    
    print(f"MCD = {mcd_values}; STOI = {stoi_values}; PESQ = {pesq_values}.")