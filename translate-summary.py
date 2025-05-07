import sys
import time
from transformers import pipeline

# 要約する間隔（行数）
SUMMARY_INTERVAL = 10  # 10行ごとに要約
# 要約の最大長と最小長
MAX_SUMMARY_LENGTH = 100
MIN_SUMMARY_LENGTH = 30

# モデルのセットアップ
print("モデルをロード中...")
try:
    # 翻訳パイプライン（日本語→英語）
    print("日英翻訳モデルをロード中...")
    translator_ja_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en", device="cpu")

    # 翻訳パイプライン（英語→日本語）
    print("英日翻訳モデルをロード中...")
    translator_en_to_ja = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ja", device="cpu")

    # 要約パイプライン（英語）
    print("要約モデルをロード中...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cpu")

    print("全てのモデルのロードが完了しました")
except Exception as e:
    print(f"モデルのロード中にエラーが発生しました: {e}")
    raise

# 要約用のバッファ
text_buffer = ""
# 行カウンター
line_count = 0

def translate_and_summarize(text):
    """
    テキストを翻訳して要約し、結果を日本語に戻す
    """
    if len(text.strip()) < 50:  # テキストが短すぎる場合は要約しない
        return "テキストが短すぎるため、要約できません。"

    try:
        # 日本語から英語に翻訳
        print("日本語テキストを英語に翻訳中...")
        translation = translator_ja_to_en(text, max_length=1024)
        english_text = translation[0]['translation_text']
        print(f"翻訳結果: {english_text[:100]}...")

        # 英語テキストを要約
        print("英語テキストを要約中...")
        summary = summarizer(english_text, max_length=MAX_SUMMARY_LENGTH, min_length=MIN_SUMMARY_LENGTH, do_sample=False)
        english_summary = summary[0]['summary_text']
        print(f"英語要約: {english_summary}")

        # 英語の要約を日本語に翻訳
        print("要約を日本語に翻訳中...")
        back_translation = translator_en_to_ja(english_summary, max_length=1024)
        japanese_summary = back_translation[0]['translation_text']

        return japanese_summary
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        return "処理中にエラーが発生しました。"

def process_text_file(file_path):
    """
    テキストファイルを読み込んで、一定間隔で要約する
    """
    global text_buffer, line_count

    print(f"テキストファイル '{file_path}' を処理します...")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            print("\nテキスト処理と要約を開始します...")
            print("=" * 50)

            for line in file:
                # 行を表示
                print(line.strip())

                # バッファに追加
                text_buffer += line
                line_count += 1

                # 一定間隔で要約
                if line_count >= SUMMARY_INTERVAL:
                    if text_buffer.strip():
                        print("\n\n=== 要約 ===")
                        summary = translate_and_summarize(text_buffer)
                        print(summary)
                        print("===========\n")

                        # バッファをクリア
                        text_buffer = ""
                        line_count = 0

                # 処理の様子を見るために少し待機
                time.sleep(0.1)

            # 最終的な要約
            if text_buffer.strip():
                print("\n\n=== 最終要約 ===")
                final_summary = translate_and_summarize(text_buffer)
                print(final_summary)
                print("===========\n")

    except Exception as e:
        print(f"テキストファイルの処理中にエラーが発生しました: {e}")
        raise

    print("\n処理が完了しました。")

if __name__ == "__main__":
    # コマンドライン引数からテキストファイルを取得
    if len(sys.argv) > 1:
        text_file = sys.argv[1]
    else:
        text_file = "transcript.txt"  # デフォルトのファイル名

    process_text_file(text_file)
