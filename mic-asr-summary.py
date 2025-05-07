import sys
import pyaudio
import numpy as np
import time
import signal
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from transformers import pipeline

# マイク入力のパラメータ設定
CHUNK = 2048          # 一度に読み込むサンプル数
FORMAT = pyaudio.paInt16  # 16ビット整数で音声を取得
CHANNELS = 1          # モノラル入力
RATE = 16000         # サンプリングレート

# ASRモデルのタグ
tag = 'eml914/streaming_conformer_asr_csj'
# 要約する間隔（秒）
SUMMARY_INTERVAL = 30  # 30秒ごとに要約
# 要約の最大長と最小長
MAX_SUMMARY_LENGTH = 100
MIN_SUMMARY_LENGTH = 30

# モデルのセットアップ
print("ASRモデルをロード中...")
d = ModelDownloader()
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
    device="cuda",
    disable_repetition_detection=True,
    decoder_text_length_limit=0,
    encoded_feat_length_limit=0
)

# 要約モデルのセットアップ
print("要約モデルをロード中...")
try:
    # より小さいモデルを使用
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    print("要約モデルのロードが完了しました")
except Exception as e:
    print(f"要約モデルのロード中にエラーが発生しました: {e}")
    raise

# 出力用の変数
prev_lines = 0
# 要約用のバッファ
transcript_buffer = ""
# 最後に要約した時間
last_summary_time = time.time()
# 要約結果
current_summary = "まだ要約はありません。"
# Ctrl+Cで終了するためのフラグ
running = True

def signal_handler(sig, frame):
    """Ctrl+Cが押された時の処理"""
    global running
    print("\n\n音声認識を終了します...")
    running = False

def progress_output(text, is_summary=False):
    """
    推論中の進捗を表示する
    ・50文字ごとに改行を挿入
    ・進捗を表示するために、前の行を上書きする
    """
    global prev_lines

    # 要約の場合は特別な表示
    if is_summary:
        sys.stderr.write("\n\n=== 要約 ===\n")
        lines = ['']
        for i in text:
            if len(lines[-1]) > 50:
                lines.append('')
            lines[-1] += i
        for line in lines:
            sys.stderr.write(line + "\n")
        sys.stderr.write("===========\n\n")
        sys.stderr.flush()
        return

    # 通常の文字起こし表示
    lines = ['']
    for i in text:
        if len(lines[-1]) > 50:
            lines.append('')
        lines[-1] += i
    for i, line in enumerate(lines):
        if i == prev_lines:
            sys.stderr.write('\n\r')
        else:
            sys.stderr.write('\r\033[B\033[K')
        sys.stderr.write(line)

    prev_lines = len(lines)
    sys.stderr.flush()

def summarize_text(text):
    """
    テキストを要約する
    """
    if len(text.strip()) < 50:  # テキストが短すぎる場合は要約しない
        return "テキストが短すぎるため、要約できません。"

    try:
        summary = summarizer(text, max_length=MAX_SUMMARY_LENGTH, min_length=MIN_SUMMARY_LENGTH, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"要約中にエラーが発生しました: {e}")
        return "要約中にエラーが発生しました。"

def main():
    """
    マイク入力からリアルタイムで音声認識と要約を行う
    """
    global transcript_buffer, last_summary_time, current_summary, running

    print("マイク入力からのリアルタイム音声認識と要約を準備中...")

    # Ctrl+Cのシグナルハンドラを設定
    signal.signal(signal.SIGINT, signal_handler)

    # PyAudioの初期化
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("マイク入力の準備が完了しました")
    except Exception as e:
        print(f"マイク入力の初期化中にエラーが発生しました: {e}")
        raise

    print("\nリアルタイム音声認識と要約を開始します。話してください...")
    print("終了するには Ctrl+C を押してください")
    print("=" * 50)

    try:
        while running:
            # マイクからの音声データを取得
            data = stream.read(CHUNK, exception_on_overflow=False)
            data = np.frombuffer(data, dtype='int16')
            # 32767は16ビットのバイナリ数の上限値であり、intをfloatに変換するための正規化に使用される
            data = data.astype(np.float16) / 32767.0

            # ASR処理
            results = speech2text(speech=data, is_final=False)
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                progress_output(text)

                # 文字起こし結果をバッファに追加
                transcript_buffer += text
            else:
                progress_output("")

            # 一定間隔で要約
            if time.time() - last_summary_time > SUMMARY_INTERVAL:
                if transcript_buffer.strip():
                    print("\n\n要約中...")
                    current_summary = summarize_text(transcript_buffer)
                    progress_output(current_summary, is_summary=True)
                    last_summary_time = time.time()

                    # バッファをクリアするか、一部を保持するかの選択
                    # ここでは簡単のためにクリア
                    transcript_buffer = ""

        # 最終結果を取得
        results = speech2text(speech=np.array([], dtype=np.float16), is_final=True)
        if results is not None and len(results) > 0:
            nbests = [text for text, token, token_int, hyp in results]
            text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
            progress_output(text)
            transcript_buffer += text

        # 最終的な要約
        if transcript_buffer.strip():
            print("\n\n最終要約を生成中...")
            final_summary = summarize_text(transcript_buffer)
            progress_output(final_summary, is_summary=True)

    except KeyboardInterrupt:
        print("\n\n音声認識を終了します...")

    finally:
        # リソースの解放
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("\n音声認識と要約を終了しました")

if __name__ == "__main__":
    main()
