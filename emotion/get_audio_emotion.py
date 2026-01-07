import os
import pickle
import torch
import pandas as pd
from kimia_infer.api.kimia import KimiAudio

# =========================
# USER SETTINGS
# =========================
MODEL_PATH = "/root/autodl-tmp/moonshotai/Kimi-Audio-7B-Instruct/"

DATASET_TO_RUN = "prolific"   # "opva" / "prolific"

PROMPT_PKLS = {
    "opva": "/root/autodl-tmp/Kimi-Audio/data/opva/questions_audio_emotion_reasoning_factors.pkl",
    "prolific": "/root/autodl-tmp/Kimi-Audio/data/prolific/questions_audio_emotion_reasoning_factors.pkl",
}

AUDIO_DIR = "/root/autodl-tmp/Kimi-Audio/data/val_data_audios"
OUTPUT_ROOT = "/root/autodl-tmp/Kimi-Audio/output_reasoning_audio"

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

# =========================
# INFERENCE
# =========================
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
# MAIN PIPELINE
# =========================
def main():
    if DATASET_TO_RUN not in PROMPT_PKLS:
        raise ValueError(f"Invalid DATASET_TO_RUN: {DATASET_TO_RUN}")

    prompt_pkl = PROMPT_PKLS[DATASET_TO_RUN]
    if not os.path.exists(prompt_pkl):
        raise FileNotFoundError(f"Prompt PKL not found: {prompt_pkl}")

    print(f"\n===== Running audio reasoning for dataset: {DATASET_TO_RUN} =====")

    df = pd.read_pickle(prompt_pkl)

    required_cols = {"participantid", "qkey", "reasoning_prompt"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Prompt PKL must contain columns {required_cols}, "
            f"but got {list(df.columns)}"
        )

    out_dir = os.path.join(OUTPUT_ROOT, DATASET_TO_RUN)
    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():
        pid = str(row["participantid"])
        qkey = str(row["qkey"])
        prompt_text = row["reasoning_prompt"]

        audio_name = f"{pid}_{qkey}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_name)

        if not os.path.exists(audio_path):
            continue

        out_path = os.path.join(out_dir, f"{pid}_{qkey}.txt")

        if os.path.exists(out_path):
            continue

        try:
            print(f"[RUN] {DATASET_TO_RUN} | {audio_name}")
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
