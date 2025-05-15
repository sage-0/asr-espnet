import sys
import wave
import numpy as np
import librosa  # Add this import for resampling
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming

tag = 'eml914/streaming_conformer_asr_csj'
audio_file = "GD-ST-A_a1.wav"

# モデルのセットアップ
d=ModelDownloader()
speech2text = Speech2TextStreaming(
    **d.download_and_unpack(tag),
    token_type=None,
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.5,
    lm_weight=0.0,
    penalty=0.0,
    nbest=1,
    device = "cuda",
    disable_repetition_detection=True,
    decoder_text_length_limit=0,
    encoded_feat_length_limit=0
)


prev_lines = 0
def progress_output(text):
    """
    推論中の進捗を表示する
    ・50文字ごとに改行を挿入
    ・進捗を表示するために、前の行を上書きする
    """
    global prev_lines
    lines=['']
    for i in text:
        if len(lines[-1]) > 50:
            lines.append('')
        lines[-1] += i
    for i,line in enumerate(lines):
        if i == prev_lines:
            sys.stderr.write('\n\r')
        else:
            sys.stderr.write('\r\033[B\033[K')
        sys.stderr.write(line)

    prev_lines = len(lines)
    sys.stderr.flush()


def recognize(wavfile):
    """
    音声ファイルを読み込んで、ASR推論を行う
    """
    with wave.open(wavfile, 'rb') as wavfile:
        ch = wavfile.getnchannels()
        bits = wavfile.getsampwidth()
        rate = wavfile.getframerate()
        nframes = wavfile.getnframes()
        buf = wavfile.readframes(-1)
        data = np.frombuffer(buf, dtype=np.int16)

    # ステレオ音声をモノラルに変換
    # if ch > 1:
    #     data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)

    # 正規化（16ビットの範囲を [-1.0, 1.0] にスケール）
    speech = data.astype(np.float32) / 32768.0

    # Resample audio to match the model's expected sample rate (16 kHz)
    target_sample_rate = 16000
    if rate != target_sample_rate:
        speech = librosa.resample(speech, orig_sr=rate, target_sr=target_sample_rate)

    sim_chunk_length = 640
    if sim_chunk_length > 0:
        for i in range(len(speech) // sim_chunk_length):
            results = speech2text(speech=speech[i * sim_chunk_length:(i + 1) * sim_chunk_length], is_final=False)
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                progress_output(nbests[0])
            else:
                progress_output("")

        results = speech2text(speech[(i + 1) * sim_chunk_length:len(speech)], is_final=True)
    else:
        results = speech2text(speech, is_final=True)
    nbests = [text for text, token, token_int, hyp in results]
    progress_output(nbests[0])

recognize(audio_file)