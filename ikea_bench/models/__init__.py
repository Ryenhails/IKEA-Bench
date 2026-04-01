"""Model loading and inference dispatch for IKEA-Bench VLM evaluation."""

from .registry import MODEL_CONFIGS, get_cache_dir


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name, cache_dir=None):
    """Load model and processor/tokenizer.

    Parameters
    ----------
    model_name : str
        Key in ``MODEL_CONFIGS`` (e.g. ``"qwen2.5-vl-7b"``).
    cache_dir : str, optional
        HuggingFace cache directory.  Falls back to ``get_cache_dir()``.

    Returns
    -------
    model : PreTrainedModel
    processor_or_tokenizer : AutoProcessor | AutoTokenizer
    family : str
    """
    from ..utils import _ensure_imports, torch as _torch_ref
    _ensure_imports()
    # After _ensure_imports(), the module-level torch is populated
    from ..utils import torch

    cfg = MODEL_CONFIGS[model_name]
    family = cfg["family"]
    _cache = cache_dir or get_cache_dir()

    if family == "qwen2.5vl":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=_cache)
        return model, processor, family

    elif family == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=_cache)
        return model, processor, family

    elif family == "qwen3vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=_cache)
        return model, processor, family

    elif family == "qwen3.5":
        # Qwen3.5 uses hybrid attention (GDN + standard); flash_attention_2
        # causes CUDA illegal memory access -- use sdpa instead
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=_cache)
        return model, processor, family

    elif family == "internvl":
        # InternVL3.5 -- custom code required
        # Small models (<= 8B): .cuda() to avoid meta tensor .item() bug
        # Large models (38B+): device_map="auto" for multi-GPU sharding
        from transformers import AutoModel, AutoTokenizer
        is_large = "38B" in cfg["model_id"] or "78B" in cfg["model_id"]
        if is_large:
            model = AutoModel.from_pretrained(
                cfg["model_id"],
                cache_dir=_cache,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map="auto",
            ).eval()
        else:
            model = AutoModel.from_pretrained(
                cfg["model_id"],
                cache_dir=_cache,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], cache_dir=_cache,
            trust_remote_code=True, use_fast=False,
        )
        return model, tokenizer, family

    elif family == "llama4":
        # Llama-4-Scout: MoE vision model, requires transformers>=4.51, flex_attention
        from transformers import Llama4ForConditionalGeneration, AutoProcessor
        model = Llama4ForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flex_attention",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=_cache)
        return model, processor, family

    elif family == "gemma3":
        from transformers import Gemma3ForConditionalGeneration, AutoProcessor
        model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=_cache)
        return model, processor, family

    elif family == "phi4":
        # Phi-4-MM: custom code, AutoModelForCausalLM, manual <|image_N|> prompt format
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(
            cfg["model_id"], cache_dir=_cache, trust_remote_code=True,
        )
        return model, processor, family

    elif family == "minicpm":
        # MiniCPM-V-4.5: custom .chat() API, use AutoTokenizer (not AutoProcessor)
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], cache_dir=_cache, trust_remote_code=True,
        )
        return model, tokenizer, family

    elif family == "molmo":
        # Molmo2: AutoModelForImageTextToText + apply_chat_template(tokenize=True, return_dict=True)
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            cfg["model_id"], cache_dir=_cache, trust_remote_code=True,
        )
        return model, processor, family

    elif family == "glm4v":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            cfg["model_id"], cache_dir=_cache, trust_remote_code=True,
        )
        return model, processor, family

    elif family == "llava_ov":
        # LLaVA-OV-1.5: Qwen-VL style, uses AutoModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            cache_dir=_cache,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            cfg["model_id"], cache_dir=_cache, trust_remote_code=True,
        )
        return model, processor, family

    else:
        raise ValueError(f"Unknown model family: {family}")


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------
def run_inference_qwen(model, processor, content, images, max_new_tokens=32,
                       family="qwen3vl"):
    """Run inference for Qwen2.5-VL, Qwen3-VL, or Qwen3.5 with interleaved content."""
    from ..utils import _ensure_imports, torch as _torch_ref
    _ensure_imports()
    from ..utils import torch

    messages = [{"role": "user", "content": content}]

    # Qwen3.5 has thinking mode ON by default -- disable it for MC evaluation
    # Qwen3-VL Thinking models: keep thinking enabled (that's the point)
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if family == "qwen3.5":
        chat_kwargs["enable_thinking"] = False
    elif family in ("qwen3vl_think", "qwen3vl_think_moe"):
        chat_kwargs["enable_thinking"] = True

    text = processor.apply_chat_template(messages, **chat_kwargs)
    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True).strip()
    return response


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448):
    """Dynamic tiling for InternVL2.5 -- adapted from model README."""
    from ..utils import _ensure_imports
    _ensure_imports()
    from ..utils import torch
    from torchvision import transforms as T

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Find best tile arrangement
    target_ratios = set()
    for n in range(1, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = (1, 1)
    best_ratio_diff = float('inf')
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_width, target_height))
    processed_blocks = []
    for i in range(best_ratio[1]):
        for j in range(best_ratio[0]):
            box = (j * image_size, i * image_size,
                   (j + 1) * image_size, (i + 1) * image_size)
            processed_blocks.append(resized.crop(box))

    # Add thumbnail
    if blocks != 1:
        thumbnail = image.resize((image_size, image_size))
        processed_blocks.append(thumbnail)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    pixel_values = [transform(block) for block in processed_blocks]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def run_inference_internvl(model, tokenizer, content, images, max_new_tokens=32):
    """Run inference for InternVL2.5 with multi-image support and interleaved content."""
    from ..utils import _ensure_imports
    _ensure_imports()
    from ..utils import torch

    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}

    if not images:
        # Text-only: join all text parts
        prompt_text = "\n".join(c["text"] for c in content if c["type"] == "text")
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=None,
            question=prompt_text,
            generation_config=generation_config,
        )
        return response.strip()

    # Process each image into tiles
    all_pixel_values = []
    num_patches_list = []
    for img in images:
        pv = _internvl_dynamic_preprocess(img, min_num=1, max_num=6)
        all_pixel_values.append(pv)
        num_patches_list.append(pv.shape[0])

    pixel_values = torch.cat(all_pixel_values, dim=0).to(
        dtype=torch.bfloat16, device=model.device
    )

    # Build interleaved prompt with <image> placeholders
    prompt_parts = []
    for c in content:
        if c["type"] == "text":
            prompt_parts.append(c["text"])
        elif c["type"] == "image":
            prompt_parts.append("<image>")
    full_prompt = "\n".join(prompt_parts)

    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=full_prompt,
        generation_config=generation_config,
        num_patches_list=num_patches_list,
    )
    return response.strip()


def run_inference_hf_generic(model, processor, content, images, max_new_tokens=32):
    """Generic HF inference with interleaved content for LLaVA-OV, Gemma3, GLM, etc."""
    from ..utils import _ensure_imports
    _ensure_imports()
    from ..utils import torch

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True).strip()
    return response


def run_inference_minicpm(model, tokenizer, content, images, max_new_tokens=512):
    """MiniCPM-V-4.5: custom .chat() with [image, image, ..., text] content."""
    # Build content list: PIL images then text
    chat_content = []
    for c in content:
        if c["type"] == "image":
            chat_content.append(c["image"])
        elif c["type"] == "text":
            chat_content.append(c["text"])
    msgs = [{"role": "user", "content": chat_content}]
    response = model.chat(msgs=msgs, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
    return response.strip()


def run_inference_molmo(model, processor, content, images, max_new_tokens=512):
    """Molmo2: apply_chat_template with tokenize=True, return_dict=True."""
    from ..utils import _ensure_imports
    _ensure_imports()
    from ..utils import torch

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


def run_inference_phi4(model, processor, content, images, max_new_tokens=512):
    """Phi-4-MM: manual <|image_N|> prompt format."""
    from ..utils import _ensure_imports
    _ensure_imports()
    from ..utils import torch

    # Build prompt with <|image_N|> placeholders
    img_idx = 0
    prompt_parts = []
    for c in content:
        if c["type"] == "image":
            img_idx += 1
            prompt_parts.append(f"<|image_{img_idx}|>")
        elif c["type"] == "text":
            prompt_parts.append(c["text"])

    prompt_text = "".join(prompt_parts)
    full_prompt = f"<|user|>{prompt_text}<|end|><|assistant|>"

    inputs = processor(
        text=full_prompt,
        images=images if images else None,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode([generated], skip_special_tokens=True)[0].strip()
    return response


def run_single(model, processor_or_tokenizer, family, content, images,
               max_new_tokens=512):
    """Dispatch to the right inference function based on model family."""
    if family in ("qwen2.5vl", "qwen3vl", "qwen3vl_moe", "qwen3.5", "llava_ov"):
        return run_inference_qwen(
            model, processor_or_tokenizer, content, images,
            max_new_tokens, family=family,
        )
    elif family == "internvl":
        return run_inference_internvl(
            model, processor_or_tokenizer, content, images, max_new_tokens,
        )
    elif family == "minicpm":
        return run_inference_minicpm(
            model, processor_or_tokenizer, content, images, max_new_tokens,
        )
    elif family == "molmo":
        return run_inference_molmo(
            model, processor_or_tokenizer, content, images, max_new_tokens,
        )
    elif family == "phi4":
        return run_inference_phi4(
            model, processor_or_tokenizer, content, images, max_new_tokens,
        )
    elif family in ("internvl_hf", "gemma3", "glm4v", "llama4"):
        return run_inference_hf_generic(
            model, processor_or_tokenizer, content, images, max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
