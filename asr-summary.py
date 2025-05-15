import sys
import wave
import numpy as np
import time
import os
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from transformers import pipeline

# ASRモデルのタグ
tag = 'eml914/streaming_conformer_asr_csj'
# 処理する音声ファイル
audio_file = "./output.wav"
# 要約する間隔（秒）
SUMMARY_INTERVAL = 30  # 30秒ごとに要約
# 要約の最大長と最小長
MAX_SUMMARY_LENGTH = 100
MIN_SUMMARY_LENGTH = 30
# 処理する最大時間（秒）- テスト用
MAX_DURATION = 60  # 60秒間だけ処理する

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
    device="cuda",  # CPUを使用
    disable_repetition_detection=True,
    decoder_text_length_limit=0,
    encoded_feat_length_limit=0
)
print(f"デバイスを CPU に設定しました")

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
        # テキストを要約（英語で出力されます）
        summary = summarizer(text, max_length=MAX_SUMMARY_LENGTH, min_length=MIN_SUMMARY_LENGTH, do_sample=False)
        return f"※英語での要約結果:\n{summary[0]['summary_text']}"
    except Exception as e:
        print(f"要約中にエラーが発生しました: {e}")

def recognize_and_summarize(wavfile):
    """
    音声ファイルを読み込んで、ASR推論と要約を行う
    """
    global transcript_buffer, last_summary_time, current_summary

    print(f"音声ファイル '{wavfile}' を処理します...")
    print(f"現在の作業ディレクトリ: {os.getcwd()}")
    print(f"ファイルが存在するか確認: {os.path.exists(wavfile)}")

    # 音声ファイルの読み込み
    try:
        with wave.open(wavfile, 'rb') as wavfile:
            ch = wavfile.getnchannels()
            bits = wavfile.getsampwidth()
            rate = wavfile.getframerate()
            nframes = wavfile.getnframes()
            buf = wavfile.readframes(-1)
            data = np.frombuffer(buf, dtype='int16')
        print(f"音声ファイルを読み込みました: チャンネル数={ch}, ビット数={bits}, サンプリングレート={rate}, フレーム数={nframes}")
    except Exception as e:
        print(f"音声ファイルの読み込み中にエラーが発生しました: {e}")
        raise

    # 音声データの正規化
    speech = data.astype(np.float16) / 32767.0

    # 音声の長さ（秒）を計算
    audio_duration = nframes / rate
    print(f"音声ファイルの長さ: {audio_duration:.2f}秒")

    # チャンクサイズの設定（16kHzの場合、640サンプルは0.04秒に相当）
    sim_chunk_length = 640

    # 音声の総サンプル数
    total_samples = len(speech)

    # 現在の処理位置（サンプル数）
    current_position = 0

    # 現在の処理位置（秒）
    current_time = 0

    print("\n音声認識と要約を開始します...")
    print("=" * 50)

    # 処理する最大サンプル数を計算（MAX_DURATION秒分）
    max_samples = min(total_samples, int(MAX_DURATION * rate))
    print(f"処理する最大時間: {MAX_DURATION}秒 ({max_samples}サンプル)")

    # チャンクごとに処理
    if sim_chunk_length > 0:
        for i in range(min(max_samples // sim_chunk_length, total_samples // sim_chunk_length)):
            # 現在の時間を更新
            current_time = (i * sim_chunk_length) / rate

            # 最大処理時間に達したら終了
            if current_time >= MAX_DURATION:
                print(f"最大処理時間 {MAX_DURATION}秒 に達したため、処理を終了します")
                break

            # ASR処理
            chunk = speech[i * sim_chunk_length:(i + 1) * sim_chunk_length]
            print(f"チャンク処理中: {i+1}/{total_samples // sim_chunk_length}, サイズ: {len(chunk)}")
            try:
                results = speech2text(speech=chunk, is_final=False)
                print(f"チャンク {i+1} の処理が完了しました")
            except Exception as e:
                print(f"チャンク {i+1} の処理中にエラーが発生しました: {e}")
                print(f"チャンクの形状: {chunk.shape}, データ型: {chunk.dtype}")
                print(f"チャンクの最小値: {chunk.min()}, 最大値: {chunk.max()}")
                raise

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

        # 残りのサンプルを処理（最大時間内のみ）
        last_processed_sample = min((i + 1) * sim_chunk_length, max_samples)
        remaining_chunk = speech[last_processed_sample:min(max_samples, total_samples)]
        if len(remaining_chunk) > 0:
            print(f"残りのチャンクを処理中: サイズ: {len(remaining_chunk)}")
            try:
                results = speech2text(speech=remaining_chunk, is_final=True)
                print("残りのチャンクの処理が完了しました")
            except Exception as e:
                print(f"残りのチャンクの処理中にエラーが発生しました: {e}")
                print(f"チャンクの形状: {remaining_chunk.shape}, データ型: {remaining_chunk.dtype}")
                print(f"チャンクの最小値: {remaining_chunk.min()}, 最大値: {remaining_chunk.max()}")
                raise
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                progress_output(text)
                transcript_buffer += text
    else:
        # 一度に全体を処理（非ストリーミングモード）- 最大時間内のみ
        print("非ストリーミングモードで全体を処理中...")
        try:
            # 最大時間分のサンプルだけを処理
            speech_to_process = speech[:max_samples]
            print(f"処理するサンプル数: {len(speech_to_process)}/{len(speech)}")
            results = speech2text(speech=speech_to_process, is_final=True)
            print("全体の処理が完了しました")
        except Exception as e:
            print(f"全体の処理中にエラーが発生しました: {e}")
            print(f"音声データの形状: {speech.shape}, データ型: {speech.dtype}")
            print(f"音声データの最小値: {speech.min()}, 最大値: {speech.max()}")
            raise
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

    print("\n処理が完了しました。")

if __name__ == "__main__":
    # コマンドライン引数から音声ファイルを取得（指定がなければデフォルト値を使用）
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]

    print(f"処理する音声ファイル: {audio_file}")
    recognize_and_summarize(audio_file)
