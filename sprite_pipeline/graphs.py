"""Builders for ComfyUI API-format graphs.

Each builder returns a dict ready for POST /prompt. Node wiring mirrors the
user's known-good workflows:
  - Krea-2 Turbo t2i        (image_krea2_turbo_t2i.json template)
  - MiniMax H3 ref-to-video (video_minimax_h3_r2v_api.json)
  - Nunchaku Qwen Image Edit 2509 Lightning (nunchaku-qwen-image-edit-2509-lightning.json)
Model filenames are the ones actually installed on this server.
"""

KREA_UNET = "krea2_turbo_fp8_scaled.safetensors"
KREA_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"

H3_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
H3_TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

# The nunchaku int4 loader crashes on this ComfyUI build (list/dtype bug in
# the custom node), so we use the core-node path: fp8 UNET + 4-step LoRA.
QWEN_EDIT_UNET = "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
QWEN_EDIT_LORA = "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_EDIT_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"


def h3_valid_length(seconds):
    """MiniMax H3 frame count must be 5 + 17k (24 fps). Round to nearest valid."""
    frames = max(5, int(round(seconds * 24)))
    k = max(1, int(round((frames - 5) / 17.0)))
    return 5 + 17 * k


def krea_t2i(prompt, width=1024, height=1024, seed=1, steps=8, cfg=1.0,
             filename_prefix="sprites/ref"):
    """Krea-2 Turbo text-to-image. 8 steps / cfg 1 / euler / simple per template."""
    return {
        "10": {"class_type": "UNETLoader",
               "inputs": {"unet_name": KREA_UNET, "weight_dtype": "default"}},
        "11": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": KREA_CLIP, "type": "krea2", "device": "default"}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["11", 0], "text": prompt}},
        "13": {"class_type": "ConditioningZeroOut",
               "inputs": {"conditioning": ["6", 0]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"model": ["10", 0], "positive": ["6", 0], "negative": ["13", 0],
                         "latent_image": ["5", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["12", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}},
    }


def minimax_h3_r2v(prompt, ref_image_names, width, height, length, seed=1,
                   use_turbo_lora=True, steps=None, fps=24.0,
                   frames_prefix="sprites/frames/frame",
                   video_prefix="sprites/video/clip",
                   save_video=True, ref_image_size="match"):
    """MiniMax H3 reference-to-video.

    ref_image_names: list of filenames already in the ComfyUI input dir
                     (referenced in the prompt as <Picture 1>, <Picture 2>, ...).
    length: frame count, must be 5 + 17k (use h3_valid_length()).
    Turbo LoRA -> 4 steps, full model -> 20 steps (per workflow defaults).
    Frames are saved as lossless PNGs; an mp4 preview is optional.
    """
    if (length - 5) % 17 != 0:
        raise ValueError("H3 length must be 5+17k, got %d" % length)
    if steps is None:
        steps = 4 if use_turbo_lora else 20
    g = {
        "127": {"class_type": "UNETLoader",
                "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
        "128": {"class_type": "CLIPLoader",
                "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
        "119": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
        "120": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
        "123": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        # Scheduler runs on the base model even when the LoRA is active (as in
        # the reference workflow).
        "124": {"class_type": "BasicScheduler",
                "inputs": {"model": ["127", 0], "scheduler": "simple",
                           "steps": steps, "denoise": 1.0}},
    }
    ref_inputs = {}
    for i, name in enumerate(ref_image_names):
        node_id = str(200 + i)
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        ref_inputs["ref_images.ref_image_%d" % i] = [node_id, 0]

    cond_inputs = {"prompt": prompt, "width": width, "height": height,
                   "length": length, "ref_image_size": ref_image_size,
                   "clip": ["128", 0], "vae": ["119", 0], "audio_vae": ["120", 0]}
    cond_inputs.update(ref_inputs)
    g["136"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}

    model_ref = ["127", 0]
    if use_turbo_lora:
        g["145"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["127", 0], "lora_name": H3_TURBO_LORA,
                               "strength_model": 1.0}}
        model_ref = ["145", 0]

    g["126"] = {"class_type": "BasicGuider",
                "inputs": {"model": model_ref, "conditioning": ["136", 0]}}
    g["125"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["129", 0], "guider": ["126", 0],
                           "sampler": ["123", 0], "sigmas": ["124", 0],
                           "latent_image": ["136", 1]}}
    g["122"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["125", 0], "vae": ["119", 0]}}
    g["150"] = {"class_type": "SaveImage",
                "inputs": {"images": ["122", 0], "filename_prefix": frames_prefix}}
    if save_video:
        g["130"] = {"class_type": "CreateVideo",
                    "inputs": {"images": ["122", 0], "fps": fps, "bit_depth": 8}}
        g["92"] = {"class_type": "SaveVideo",
                   "inputs": {"video": ["130", 0], "filename_prefix": video_prefix,
                              "format": "auto", "codec": "auto"}}
    return g


def qwen_edit(prompt, image_name, seed=1, steps=4, cfg=1.0, megapixels=1.0,
              filename_prefix="sprites/edit", extra_image_names=None):
    """Nunchaku Qwen Image Edit 2509 Lightning (4 steps, cfg 1).

    image_name: filename in the ComfyUI input dir (the image being edited);
    extra_image_names: up to 2 additional reference images (image2/image3).
    """
    extra_image_names = extra_image_names or []
    g = {
        "115": {"class_type": "UNETLoader",
                "inputs": {"unet_name": QWEN_EDIT_UNET, "weight_dtype": "default"}},
        "116": {"class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["115", 0], "lora_name": QWEN_EDIT_LORA,
                           "strength_model": 1.0}},
        "38": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": QWEN_EDIT_CLIP, "type": "qwen_image",
                          "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "66": {"class_type": "ModelSamplingAuraFlow",
               "inputs": {"model": ["116", 0], "shift": 3.0}},
        "75": {"class_type": "CFGNorm", "inputs": {"model": ["66", 0], "strength": 1.0}},
        "78": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "93": {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"image": ["78", 0], "upscale_method": "lanczos",
                          "megapixels": megapixels, "resolution_steps": 1}},
        "88": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["93", 0], "vae": ["39", 0]}},
    }
    pos_inputs = {"clip": ["38", 0], "vae": ["39", 0], "prompt": prompt,
                  "image1": ["93", 0]}
    neg_inputs = {"clip": ["38", 0], "vae": ["39", 0], "prompt": "",
                  "image1": ["93", 0]}
    for i, name in enumerate(extra_image_names[:2]):
        node_id = str(300 + i)
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        pos_inputs["image%d" % (i + 2)] = [node_id, 0]
        neg_inputs["image%d" % (i + 2)] = [node_id, 0]
    g["111"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": pos_inputs}
    g["110"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": neg_inputs}
    g["3"] = {"class_type": "KSampler",
              "inputs": {"model": ["75", 0], "positive": ["111", 0],
                         "negative": ["110", 0], "latent_image": ["88", 0],
                         "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}}
    g["60"] = {"class_type": "SaveImage",
               "inputs": {"images": ["8", 0], "filename_prefix": filename_prefix}}
    return g
