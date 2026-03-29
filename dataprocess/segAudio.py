import os
from scipy.io import wavfile
import librosa
import numpy as np
def seg_song_preprocess(input_path, outputFilePath, segment_duration, target_sr, basefilename):
    """
    Divide the long audio into smaller segments, and perform the division in a sliding window manner.

    input_file: audio path
    outputFilePath: save path
    segment_duration: The duration of each audio segment is measured in seconds.
    """

    # load audio
    audio, fs = librosa.load(input_path)

    if fs != target_sr:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_sr)
    audio_length = len(audio) / target_sr
    step_value = 1

    segment_samples = int(segment_duration * target_sr)
    slide_samples = int(target_sr*step_value)

    if len(audio) >= segment_samples:

        segment_count = 0
        for start_sample in range(0, len(audio) - segment_samples + 1, slide_samples):

            end_sample = start_sample + segment_samples

            segment_audio = audio[start_sample:end_sample]

            segment_count += 1
            name_part = f'{basefilename}_seg{segment_count}.wav'
            output_file = os.path.join(outputFilePath, name_part)

            wavfile.write(output_file, target_sr, segment_audio)
            print(f'Saved segment: {output_file}')
        print(f'Total segments saved: {segment_count}')
    else:
        padded_audio = np.zeros(segment_samples, dtype=np.float32)
        padded_audio[:len(audio)] = audio
        name_part = f'{basefilename}.wav'
        output_file = os.path.join(outputFilePath, name_part)
        wavfile.write(output_file, target_sr, padded_audio)
        print(f'Saved segment: {output_file}')


input_file = r'../hainangibbon'
output_dir = r'../hainangibbonseg'
audio_extention = ['.wav', '.mp3']
segment_duration = 10
target_sr = 16000

for root, dirs, files in os.walk(input_file):
    for dir_name in dirs:
        dirfilesPath = os.path.join(root, dir_name)
        dirfiles = os.listdir(dirfilesPath)
        outputFilePath = os.path.join(output_dir, dir_name)
        for file in dirfiles:
            if any(file.lower().endswith(ext) for ext in audio_extention):
                input_path = os.path.join(dirfilesPath, file)
                basefilename = os.path.splitext(file)[0]

                if not os.path.exists(outputFilePath):
                    os.makedirs(outputFilePath)
                try:
                    seg_song_preprocess(input_path, outputFilePath, segment_duration, target_sr, basefilename)
                except Exception as e:
                    print(f"Critical error with {input_path}: {e}")
                    continue
