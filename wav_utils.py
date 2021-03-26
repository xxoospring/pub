import os
import wave
import numpy as np
import scipy.io.wavfile
import librosa
import wavio
from scipy.signal import resample as sci_resample
from pysndfx import AudioEffectsChain
import pydub
import pyaudio


def wav_read(wav_file):
    assert wav_file.endswith('wav'), "file error"
    try:
        sr, data = scipy.io.wavfile.read(wav_file)
    except:
        WAV = wavio.read(wav_file)
        sr, data = WAV.rate, np.squeeze(WAV.data)
    return sr, data


def wav_write(wav_file, data, src_sr, dest_sr=None):
    if not dest_sr:
        scipy.io.wavfile.write(wav_file, src_sr, data.astype(np.int16))
    else:
        re_data = librosa.resample(data.astype(np.float32), src_sr, dest_sr)
        scipy.io.wavfile.write(wav_file, dest_sr, re_data.astype(np.int16))


def wav_check(file_path, sr, nch, samplebits):
    f = wave.open(file_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    f.close()
    if nch != nchannels:
        print('Channel not Match!')
        return False
    if sr != framerate:
        print('SR not Match!')
        return False
    if samplebits != sampwidth:
        print('Bits not Match!')
        return False

def wave2pcm(in_file, out_file):
    assert out_file.endswith('.pcm') and in_file.endswith('.wav'), "Out Format Error"
    _, data = wav_read(in_file)
    with open(out_file, 'wb') as pcm:
        pcm.write(data.astype(np.int16).tostring())


def pcm2wav(pcm_file, wav_file, ch=1, bytes=2, sr=8000):
    data = np.fromfile(pcm_file, np.short)
    wav_write(wav_file, data, sr)


def resample(arr, sr_src, sr_dest):
    if sr_src == sr_dest:
        return arr
    # return librosa.resample(arr, sr_src, sr_dest) # realy slow
    n = arr.shape[0]
    n_target = int(n * sr_dest / sr_src)
    return sci_resample(arr, n_target)

def normalize_volume(wav_args, vol_dest=-20, in_place=True):
    assert type(wav_args) in [list, str], 'not support format'
    tmp = []
    if type(wav_args) == str:
        tmp.append(wav_args)
    else:
        tmp = wav_args
    for it in tmp:
        sd = pydub.AudioSegment.from_wav(it)
        vol_delta = vol_dest-sd.dBFS
        ret = sd + vol_delta
        if in_place:
            ret.export(it, format='wav')
        else:
            path, name = os.path.split(it)
            ret.export(os.path.join(path, "%sdB-"%vol_dest+name), format='wav')


class VoicePostHandle(AudioEffectsChain):
    def __init__(self):
        self.command = []

    def clear_command(self):
        self.command = []

    @staticmethod
    def _voice_volume_setting(p=50.):
        if p < 0:
            return 0.
        if p > 100:
            return 2.
        return 1 + (p - 50) / 50.

    def volume_adjust(self, wave_arr, p=50.):
        factor = self._voice_volume_setting(p)
        if factor == 1.:
            return wave_arr.astype(np.int16)
        else:
            return (wave_arr * factor).astype(np.int16)

    @staticmethod
    # shift gives the pitch shift as positive or negative ‘cents’ (i.e. 100ths of a semitone)
    # -1200~1200, 12 semitones
    def _voice_pitch_setting(p=50.):
        if p < 0:
            return -1200.
        if p > 100:
            return 1200.
        return 24. * (p - 50)

    def pitch_adjust(self, wav_arr, p=50., sample_arte=8000):
        factor = self._voice_pitch_setting(p)
        if factor == 0.:
            return wav_arr.astype(np.int16)
        else:
            self.pitch(factor)
            out_wav_arr = self(wav_arr.astype(np.int16), sample_in=sample_arte)
            self.clear_command()
            return out_wav_arr.astype(np.int16)

    @staticmethod
    def _voice_speed_setting(p=50.):
        if p <= 5:  # sox -tempo [0.1 < fator < 100]
            return 0.11
        if p > 100:
            return 2.
        return 1 + (p - 50) / 50.

    def speed_adjust(self, wav_arr, p=50., sample_arte=8000):
        factor = self._voice_speed_setting(p)
        if factor == 1.:
            return wav_arr.astype(np.int16)
        else:
            self.tempo(factor)
            out_wav_arr = self(wav_arr.astype(np.int16), sample_in=sample_arte)
            self.clear_command()
            return out_wav_arr.astype(np.int16)


def play(fname, speed=1.0):
    chunk = 1024  # 2014kb
    wf = wave.open(fname, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=int(wf.getframerate()*speed), output=True)

    data = wf.readframes(chunk)  # 读取数据
    # print(data)

    def avg_amp_adjust(src_y, level=1500.):
        k = level / (np.average(np.abs(src_y)))
        dest = src_y * k
        dest[np.abs(dest) > 32767] = 32767
        return dest

    while data:  # 播放
        # print(type(data)) # bytes
        # data = str(data)
        stream.write(data)
        data = wf.readframes(chunk)

        # print('while循环中！')
        # print(data)
    stream.stop_stream()  # 停止数据流
    stream.close()
    p.terminate()  # 关闭 PyAudio


def online_play(nd_data, sr):
    p = pyaudio.PyAudio()

    # open stream (2), 2 is size in bytes of int16
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=sr,
                    output=True)

    # play stream (3), blocking call
    stream.write(nd_data.astype(np.short).tostring())

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    p.terminate()


if __name__ == '__main__':
    # wav_check('../test.wav', 16000, 2, 2)
    # play('../test.wav', 2.0)
    data = wav_read('../test.wav')[1]

