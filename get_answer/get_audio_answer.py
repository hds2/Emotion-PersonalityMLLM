import os
import torch
import pandas as pd
from kimia_infer.api.kimia import KimiAudio

# =========================
# USER SETTINGS
# =========================
MODEL_PATH = "/root/autodl-tmp/moonshotai/Kimi-Audio-7B-Instruct/"

DATASET = "prolific"   # "opva" / "prolific"


PROMPT_PKL = "/root/autodl-tmp/Kimi-Audio/personality_prompt/questions_personality_audio_prolific.pkl"

AUDIO_DIR = "/root/autodl-tmp/Kimi-Audio/data/data_audios"
OUTPUT_ROOT = f"/root/autodl-tmp/Kimi-Audio/output_personality/{DATASET}"

# =========================
# LOAD MODEL
# =========================
model = KimiAudio(model_path=MODEL_PATH, load_detokenizer=True)

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

def run_audio_inference(audio_path: str, prompt_text: str) -> str:
    messages = [
        {"role": "user", "message_type": "text", "content": prompt_text},
        {"role": "user", "message_type": "audio", "content": audio_path},
    ]
    _, text_output = model.generate(
        messages,
        **sampling_params,
        output_type="text",
    )
    return text_output or ""

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    
    df = pd.read_pickle(PROMPT_PKL)

    required_cols = {
        "participantid",
        "qkey",
        "activated_dimension",
        "personality_prompt",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Prompt PKL must contain columns {required_cols}, "
            f"but got {list(df.columns)}"
        )

   
    for _, row in df.iterrows():
        pid = str(row["participantid"])
        qkey = str(row["qkey"])
        dimension = str(row["activated_dimension"])
        prompt_text = row["personality_prompt"]

        audio_name = f"{pid}_{qkey}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_name)

        if not os.path.exists(audio_path):
            continue

        dim_dir = os.path.join(OUTPUT_ROOT, dimension)
        os.makedirs(dim_dir, exist_ok=True)

        out_path = os.path.join(dim_dir, f"{pid}_{qkey}.txt")
        if os.path.exists(out_path):
            continue

        try:
            print(f"[RUN] {pid}_{qkey} | {dimension}")
            result = run_audio_inference(audio_path, prompt_text)

            if result.strip():
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"[SAVE] {out_path}")
            else:
                print(f"[EMPTY] {audio_name}")

        except Exception as e:
            print(f"[ERROR] {audio_name}: {e}")

if __name__ == "__main__":
    main()
