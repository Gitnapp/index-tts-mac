from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice="./test_data/input.wav"
with open("./payload.txt", "r", encoding="utf-8") as f:
    text = f.read()
output_path="output.wav"
tts.infer(voice, text, output_path)