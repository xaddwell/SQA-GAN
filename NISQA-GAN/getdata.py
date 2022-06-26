import os
import h5py
import scipy
import librosa
import numpy as np
from tqdm import tqdm
# import tensorflow as tf
from os.path import join
import random
import soundfile as sf
random.seed(1984)
FS = 16000

#withReverberation
# with_DATA_DIR = r'D:\cjh\TencentCorups'
# with_AUDIO_DIR = join(with_DATA_DIR, 'withReverberationTrainDev')

#withoutReverberation
without_DATA_DIR = r'D:\cjh\val\TUB_IS22_DB1'
without_AUDIO_DIR = join(without_DATA_DIR, 'TUB_IS22_DB1')

#pstn
# pstn_DATA_DIR = r'D:\cjh\train'
# pstn_AUDIO_DIR = join(pstn_DATA_DIR, 'pstn_train')


def get_spectrograms(sound_file,enhanced_file,fs=FS):
    # Loading sound file
    y, _ = librosa.load(sound_file,sr=fs)  # or set sr to hp.sr.
    y1, _ = librosa.load(enhanced_file, sr=fs)
    diff_y=y1-y
    concat_y=np.concatenate((y,diff_y))
    return concat_y

def extract_to_npy():
    # with_audio_dir = with_AUDIO_DIR
    without_audio_dir = without_AUDIO_DIR
    # pstn_audio_dir = pstn_AUDIO_DIR

    # print('with_audio_dir dir: {}'.format(with_audio_dir))
    print('without_audio_dir: {}'.format(without_audio_dir))
    # print('pstn_audio_dir: {}'.format(pstn_audio_dir))

    # get filenames
    files = []
    for f in os.listdir(without_audio_dir):
        if f.endswith('.wav'):
            files.append(join(without_audio_dir,f.split('\\')[-1]))

    for i in tqdm(range(len(files))):
        f = files[i]
        enhanced_f=join(without_DATA_DIR,"TUB_ENHANCED/"+f.split('\\')[-1])
        print(f,enhanced_f)
        concat_y = get_spectrograms(f,enhanced_f)
        dir=r"D:\cjh\val\TUB_IS22_DB1\TUB_concat"
        sf.write(join(dir,f.split('\\')[-1]), concat_y, FS)

if __name__ == '__main__':
    extract_to_npy()