import os
import torch
import pandas as pd
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =========================
# USER SETTINGS
# =========================
HF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

DATASET = "opva"   # "opva" / "prolific"

PROMPT_PKL = r"D:\code\code-Experiment1\personality_prompt\questions_personality_video_opva.pkl"
# PROMPT_PKL = r"D:\code\code-Experiment1\personality_prompt\questions_personality_video_prolific.pkl"

VIDEO_DIR = r"D:\code\code-Experiment1\data\video"
OUTPUT_ROOT = r"D:\code\code-Experiment1\output_personality_video"

MAX_PIXELS = 360 * 420
FPS = 1.0
MAX_NEW_TOKENS = 4096

# =========================
# MODEL (load once)
# =========================
_model = None
_processor = None


def load_model_and_processor():
    global _model, _processor
    if _model is not None:
        return

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    _model.eval()
    _processor = AutoProcessor.from_pretrained(HF_MODEL_ID)


def get_device():
    if torch.cuda.is_available():
        return next(_model.parameters()).device
    return torch.device("cpu")


@torch.no_grad()
def run_video_inference(video_path: str, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": MAX_PIXELS,
                    "fps": FPS,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    inputs = inputs.to(get_device())

    outputs = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    outputs = outputs[:, inputs.input_ids.shape[1]:]

    decoded = _processor.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return decoded[0] if decoded else ""


# =========================
# MAIN PIPELINE
# =========================
def main():
    load_model_and_processor()

    if not os.path.exists(PROMPT_PKL):
        raise FileNotFoundError(f"Prompt PKL not found: {PROMPT_PKL}")

    print(f"\n===== Running VIDEO PERSONALITY inference: {DATASET} =====")

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

    out_root = os.path.join(OUTPUT_ROOT, DATASET)
    os.makedirs(out_root, exist_ok=True)

    for _, row in df.iterrows():
        pid = str(row["participantid"])
        qkey = str(row["qkey"])
        dimension = str(row["activated_dimension"])
        prompt_text = row["personality_prompt"]

        video_name = f"{pid}_{qkey}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_name)

        if not os.path.exists(video_path):
            continue

        dim_dir = os.path.join(out_root, dimension)
        os.makedirs(dim_dir, exist_ok=True)

        out_path = os.path.join(dim_dir, f"{pid}_{qkey}.txt")
        if os.path.exists(out_path):
            continue

        try:
            print(f"[RUN] {pid}_{qkey} | {dimension}")
            result = run_video_inference(video_path, prompt_text)

            if result.strip():
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"[SAVE] {out_path}")
            else:
                print(f"[EMPTY] {video_name}")

        except Exception as e:
            print(f"[ERROR] {video_name}: {e}")


if __name__ == "__main__":
    main()
