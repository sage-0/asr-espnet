import sys
import time
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# 要約する間隔（行数）
SUMMARY_INTERVAL = 10  # 10行ごとに要約
# 要約の最大長と最小長
MAX_SUMMARY_LENGTH = 100
MIN_SUMMARY_LENGTH = 30

# 要約モデルのセットアップ
print("要約モデルをロード中...")
try:
    # 多言語対応のモデルを使用
    # mT5モデルは多言語テキスト要約に対応しています
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # CPUを使用するように設定
    device = "cpu"
    print(f"デバイスを {device} に設定しました")

    # 要約関数の定義
    def generate_summary(text):
        # mT5モデルでは、タスクのプレフィックスを追加する必要があります
        prefix = "summarize: "
        input_text = prefix + text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        model.to(device)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=MAX_SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    print("要約モデルのロードが完了しました")
except Exception as e:
    print(f"要約モデルのロード中にエラーが発生しました: {e}")
    raise

# 要約用のバッファ
text_buffer = ""
# 行カウンター
line_count = 0

def summarize_text(text):
    """
    テキストを要約する
    """
    if len(text.strip()) < 50:  # テキストが短すぎる場合は要約しない
        return "テキストが短すぎるため、要約できません。"

    try:
        summary = generate_summary(text)
        return summary
    except Exception as e:
        print(f"要約中にエラーが発生しました: {e}")
        return "要約中にエラーが発生しました。"

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
                        summary = summarize_text(text_buffer)
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
                final_summary = summarize_text(text_buffer)
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
