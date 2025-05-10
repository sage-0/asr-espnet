import sys
from reazonspeech.k2.asr import load_model, transcribe, audio_from_path

# GPUで推論したい場合は device='cuda' と指定ください
model = load_model(device='cpu')

# ローカルの音声ファイルを読み込む
audio = audio_from_path(sys.argv[1])

# 音声認識を適用する
ret = transcribe(model, audio)

print(ret.text)