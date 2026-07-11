from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


def load_model(model_path: str):
    model, processor = load(model_path)
    config = load_config(model_path)
    return model, processor, config


def run(
    model,
    processor,
    config,
    image_path: str,
    prompt: str = "Describe this image in detail.",
) -> str:
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    output = generate(
        model,
        processor,
        formatted_prompt,
        image=[image_path],
        max_tokens=256,
        temperature=0.0,
        verbose=False,
    )

    if isinstance(output, str):
        return output
    return getattr(output, "text", str(output))
