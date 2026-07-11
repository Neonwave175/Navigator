import time as t
import warnings

from mlx_vlm.generate import generate
from mlx_vlm.utils import load

warnings.filterwarnings("ignore", message=".*mel filter.*")

# Load
start = t.perf_counter()
model, processor = load("mlx-community/gemma-4-26b-a4b-it-mxfp4")
print(f"Load: {t.perf_counter() - start:.2f}s\n")

image_path = "images/Newt.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image"},
        ],
    }
]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Warmup — first call eats Metal shader compilation + JIT cost, skews real numbers
_ = generate(
    model,
    processor,
    prompt,
    image=image_path,
    max_tokens=16,
    temperature=0.0,
    verbose=False,
)

# Timed runs
n_runs = 3
for i in range(n_runs):
    start = t.perf_counter()
    result = generate(
        model,
        processor,
        prompt,
        image=image_path,
        max_tokens=256,
        temperature=0.0,
        max_kv_size=4096,  # cap context — no need for 256K on single-image inference
        verbose=False,
    )
    elapsed = t.perf_counter() - start
    print(f"Run {i}: wall={elapsed:.2f}s  {result}")

print("\n" + result.text)
