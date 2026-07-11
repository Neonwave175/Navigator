import warnings

from mlx_vlm.generate import generate
from mlx_vlm.utils import load

warnings.filterwarnings("ignore", message=".*mel filter.*")

model, processor = load("mlx-community/gemma-4-26b-a4b-it-mxfp4")

# This is the location of the image, edit this if needed
image_path = "images/Linus.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Describe this image",
            },
        ],
    }
]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
result = generate(
    model,
    processor,
    prompt,
    image=image_path,  # Multiple images can be given
    max_tokens=256,
    temperature=0.0,
)
print(result.text)
