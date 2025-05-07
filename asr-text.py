import sys
import pyaudio
import numpy as np
import signal
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming

# マイク入力のパラメータ設定
CHUNK=2048          # 一度に読み込むサンプル数
FORMAT=pyaudio.paInt16  # 16ビット整数で音声を取得
CHANNELS=1          # モノラル入力
RATE=16000         # サンプリングレート

tag = 'eml914/streaming_conformer_asr_csj'

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


p=pyaudio.PyAudio()
stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)

# Ctrl+Cで終了するためのフラグ
running = True

def signal_handler(sig, frame):
    """Ctrl+Cが押された時の処理"""
    global running
    print("\n\n音声認識を終了します...")
    running = False

# Ctrl+Cのシグナルハンドラを設定
signal.signal(signal.SIGINT, signal_handler)

print("\nリアルタイム音声認識を開始します。話してください...")
print("終了するには Ctrl+C を押してください")
print("=" * 50)

try:
    while running:
        data = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(data, dtype='int16')
        # 32767は16ビットのバイナリ数の上限値であり、intをfloatに変換するための正規化に使用される
        data = data.astype(np.float16)/32767.0

        results = speech2text(speech=data, is_final=False)
        if results is not None and len(results) > 0:
            nbests = [text for text, token, token_int, hyp in results]
            text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
            progress_output(nbests[0])
        else:
            progress_output("")

    # 最終結果を取得
    results = speech2text(speech=np.array([], dtype=np.float16), is_final=True)
    if results is not None and len(results) > 0:
        nbests = [text for text, token, token_int, hyp in results]
        progress_output(nbests[0])

except KeyboardInterrupt:
    print("\n\n音声認識を終了します...")

finally:
    # リソースの解放
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("\n音声認識を終了しました")
