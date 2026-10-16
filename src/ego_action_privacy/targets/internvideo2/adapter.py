"""Frozen InternVideo2-CLIP-1B with differentiable RGB input and strict loading.

Input: RGB [B, T, 3, H, W], uint8 or floating point in [0, 1].
No external flash-attn, DeepSpeed, timm, or official package import is used.

Adapted from the separately validated SDPA compatibility implementation in the
predecessor workspace. See the repository NOTICE-InternVideo and license.
"""
from __future__ import annotations

import gc
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .vendor.vision import InternVideo2


def read_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    for name in ("module", "model", "state_dict"):
        if isinstance(checkpoint.get(name), Mapping):
            return checkpoint[name]
    return checkpoint


def interpolate_temporal_position(position, frames: int, spatial_tokens=256):
    if position.ndim != 3 or position.shape[0] != 1:
        raise ValueError(f"Invalid positional embedding: {position.shape}")
    count = position.shape[1] - 1
    if count % spatial_tokens:
        raise ValueError("Positional embedding does not match the spatial grid")
    old_frames = count // spatial_tokens
    if old_frames == frames:
        return position
    width = position.shape[-1]
    extra, tokens = position[:, :1], position[:, 1:]
    tokens = tokens.reshape(1, old_frames, spatial_tokens, width)
    tokens = tokens.permute(0, 2, 3, 1).reshape(-1, width, old_frames)
    # Same linear interpolation layout as the official pos_embed.py.
    tokens = F.interpolate(tokens.float(), size=frames, mode="linear", align_corners=False)
    tokens = tokens.to(position.dtype).reshape(1, spatial_tokens, width, frames)
    tokens = tokens.permute(0, 3, 1, 2).reshape(1, frames * spatial_tokens, width)
    return torch.cat((extra, tokens), dim=1)


def remap_text_key(key: str, expected: set[str]) -> str:
    """PEFT 0.3-style q/v weights -> PEFT 0.13 base_layer names."""
    if key in expected:
        return key
    candidate = re.sub(r"\.(q_proj|v_proj)\.(weight|bias)$", r".\1.base_layer.\2", key)
    return candidate if candidate in expected else key


def load_complete(module: nn.Module, state: Mapping, label: str):
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatches = [
        (key, tuple(state[key].shape), tuple(expected[key].shape))
        for key in expected.keys() & state.keys()
        if state[key].shape != expected[key].shape
    ]
    if missing or unexpected or mismatches:
        raise RuntimeError(
            f"{label}: missing={missing[:20]}, unexpected={unexpected[:20]}, "
            f"shape_mismatches={mismatches[:10]}"
        )
    module.load_state_dict(state, strict=True, assign=True)
    meta = [name for name, p in module.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"{label}: parameters still on meta device: {meta[:10]}")
    return {"loaded_keys": len(expected), "missing_keys": [], "unexpected_keys": []}


def build_vision(*, num_frames=8, gradient_checkpointing=False, **overrides):
    """overrides are for tiny numerical tests, not the 1B checkpoint."""
    settings = dict(
        in_chans=3, patch_size=14, img_size=224, qkv_bias=False,
        drop_path_rate=0.3, head_drop_path_rate=0.0,
        embed_dim=1408, num_heads=16, mlp_ratio=48 / 11,
        init_values=0.1, qk_normalization=True, depth=40,
        use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False,
        attn_pool_num_heads=16, clip_embed_dim=768,
        layerscale_no_force_fp32=True, num_frames=num_frames,
        tubelet_size=1, sep_pos_embed=False,
        use_checkpoint=gradient_checkpointing,
        checkpoint_num=40 if gradient_checkpointing else 0,
    )
    settings.update(overrides)
    model = InternVideo2(**settings)
    # The official fused MLP defaults to gelu_approx (tanh GELU).
    for block in model.blocks:
        block.mlp.act = nn.GELU(approximate="tanh")
    return model


class TextTower(nn.Module):
    def __init__(self, transformer, hidden_size, embedding_dim=768):
        super().__init__()
        self.transformer = transformer
        self.text_projection = nn.Parameter(torch.empty(hidden_size, embedding_dim))

    def forward(self, input_ids):
        # Preserve the official token > 0 mask and final nonpadding-token pooling.
        mask = input_ids > 0
        hidden = self.transformer(
            input_ids=input_ids, attention_mask=mask,
            use_cache=False, return_dict=True,
        ).last_hidden_state
        last = mask.long().sum(1) - 1
        if (last < 0).any():
            raise ValueError("Empty text sequence")
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return pooled @ self.text_projection


class InternVideo2CLIP(nn.Module):
    def __init__(self, model_dir, *, device="cuda", dtype=torch.bfloat16,
                 gradient_checkpointing=False, load_text_encoder=True):
        super().__init__()
        from accelerate import init_empty_weights
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

        root = Path(model_dir)
        files = dict(
            vision=root / "InternVideo2-stage2_1b-224p-f4.pt",
            text=root / "internvl_c_13b_224px.pth",
            clip=root / "1B_clip.pth",
        )
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        self.num_frames = 8
        self.image_size = 224
        self.embedding_dim = 768
        config_path = root / "chinese_alpaca_lora_7b"
        config = LlamaConfig.from_pretrained(config_path, local_files_only=True)
        if (config.hidden_size, config.num_hidden_layers, config.vocab_size) != (4096, 32, 49954):
            raise ValueError("Text configuration does not match the official CLIP-1B checkpoint")
        if config.rope_scaling:
            raise ValueError("This checkpoint requires the original unscaled RoPE")
        config.use_cache = False
        config._attn_implementation = "sdpa"
        config.causal = True

        self.tokenizer = LlamaTokenizer.from_pretrained(
            config_path, local_files_only=True, legacy=False
        )
        self.tokenizer.pad_token = " "
        self.tokenizer.add_eos_token = True
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id != 0:
            raise ValueError("Official text>0 mask requires pad_token_id=0")
        if len(self.tokenizer) != config.vocab_size:
            raise ValueError("Tokenizer/config vocabulary mismatch")

        print("[1/4] Building and loading vision encoder", flush=True)
        with init_empty_weights():
            self.vision_encoder = build_vision(
                gradient_checkpointing=gradient_checkpointing
            )
        vision_base = read_state(files["vision"])
        clip_state = read_state(files["clip"])
        vision_keys = set(self.vision_encoder.state_dict())
        vision_state = {
            key[len("vision_encoder."):]: value
            for key, value in vision_base.items()
            if key.startswith("vision_encoder.") and key[len("vision_encoder."):] in vision_keys
        }
        if "pos_embed" not in vision_state:
            raise RuntimeError("Vision checkpoint has no positional embedding")
        source_tokens = vision_state["pos_embed"].shape[1]
        vision_state["pos_embed"] = interpolate_temporal_position(
            vision_state["pos_embed"], self.num_frames
        )
        overlay_count = 0
        for key, value in clip_state.items():
            if key.startswith("vision_encoder."):
                short = key[len("vision_encoder."):]
                if short not in vision_keys:
                    raise RuntimeError(f"Unknown CLIP vision parameter: {key}")
                vision_state[short] = value
                overlay_count += 1
        if not overlay_count:
            raise RuntimeError("CLIP vision projection parameters are missing")
        self.load_report = {"vision": load_complete(self.vision_encoder, vision_state, "vision")}
        self.load_report["vision"].update(
            source_position_tokens=source_tokens,
            target_position_tokens=vision_state["pos_embed"].shape[1],
            clip_overlay_keys=overlay_count,
        )
        del vision_state, vision_base

        if not load_text_encoder:
            if "temp" not in clip_state or clip_state["temp"].numel() != 1:
                raise RuntimeError("CLIP temperature is missing")
            temperature = clip_state["temp"].float().reshape(())
            if not torch.isfinite(temperature).item() or temperature.item() <= 0:
                raise RuntimeError("Invalid CLIP temperature")
            self.register_buffer("temperature", temperature.clone())
            self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
            self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
            del self.tokenizer, clip_state
            gc.collect()
            print(f"[2/2] Moving frozen vision model to {device} ({dtype})", flush=True)
            self.to(device=device)
            self.vision_encoder.to(dtype=dtype)
            self.requires_grad_(False)
            self.eval()
            self.load_report["temperature"] = float(self.temperature.float().item())
            self.load_report["text"] = {"loaded": False, "reason": "fixed cached features requested"}
            return

        print("[2/4] Building and loading text encoder", flush=True)
        with init_empty_weights():
            language_model = LlamaForCausalLM(config)
            peft_model = get_peft_model(language_model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=True,
                target_modules=["q_proj", "v_proj"],
                r=16, lora_alpha=32, lora_dropout=0.1,
            ))
            self.text_encoder = TextTower(peft_model.base_model.model.model, 4096)
        del peft_model, language_model
        text_base = read_state(files["text"])
        raw_text = {
            key: value for key, value in text_base.items()
            if key.startswith("transformer.") or key == "text_projection"
        }
        for key, value in clip_state.items():
            if key.startswith("text_encoder."):
                raw_text[key[len("text_encoder."):]] = value
            elif key != "temp" and not key.startswith("vision_encoder."):
                raise RuntimeError(f"Unknown CLIP overlay state: {key}")

        expected = set(self.text_encoder.state_dict())
        mapped, rotary = {}, {}
        remapped_count = 0
        for key, value in raw_text.items():
            if re.fullmatch(r"transformer\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq", key):
                rotary[key] = value
                continue
            target = remap_text_key(key, expected)
            if target in mapped:
                raise RuntimeError(f"Duplicate mapped parameter: {target}")
            mapped[target] = value
            remapped_count += int(target != key)
        self.load_report["text"] = load_complete(self.text_encoder, mapped, "text")
        if len(rotary) != config.num_hidden_layers:
            raise RuntimeError(f"Expected 32 historical RoPE buffers, found {len(rotary)}")
        frequency = next(iter(rotary.values()))
        if frequency.shape != (64,) or not all(torch.equal(frequency, x) for x in rotary.values()):
            raise RuntimeError("Historical RoPE buffers differ between layers")
        # HF 4.45 centralizes RoPE rather than storing one persistent buffer per layer.
        self.text_encoder.transformer.rotary_emb.inv_freq = frequency.clone()
        self.text_encoder.transformer.rotary_emb.original_inv_freq = frequency.clone()
        self.load_report["text"].update(
            remapped_base_weights=remapped_count,
            historical_rope_buffers_preserved=len(rotary),
            lora_keys=sum(".lora_" in key for key in mapped),
        )
        if "temp" not in clip_state or clip_state["temp"].numel() != 1:
            raise RuntimeError("CLIP temperature is missing")
        temperature = clip_state["temp"].float().reshape(())
        if not torch.isfinite(temperature).item() or temperature.item() <= 0:
            raise RuntimeError("Invalid CLIP temperature")
        self.register_buffer("temperature", temperature.clone())
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        del mapped, raw_text, text_base, rotary, clip_state
        gc.collect()

        print(f"[3/4] Moving frozen model to {device} ({dtype})", flush=True)
        self.to(device=device)
        self.vision_encoder.to(dtype=dtype)
        self.text_encoder.to(dtype=dtype)
        self.requires_grad_(False)
        self.eval()
        self.load_report["temperature"] = float(self.temperature.float().item())
        print("[4/4] Required model state loaded with zero missing keys", flush=True)

    def train(self, mode=True):
        # This is a frozen attack target; U-Net training must not enable target dropout.
        return super().train(False)

    def preprocess(self, video):
        if video.ndim != 5 or video.shape[1:3] != (self.num_frames, 3):
            raise ValueError(f"Expected [B,8,3,H,W], received {tuple(video.shape)}")
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        elif not video.is_floating_point():
            raise TypeError("Video must be uint8 or floating-point RGB in [0,1]")
        if not torch.isfinite(video).all().item():
            raise ValueError("Video contains NaN/Inf")
        lo, hi = video.detach().aminmax()
        if lo.item() < 0 or hi.item() > 1:
            raise ValueError("Float video must be in [0,1]; normalize only inside this wrapper")
        video = video.to(device=self.rgb_mean.device, dtype=torch.float32)
        batch, frames, channels, height, width = video.shape
        video = F.interpolate(
            video.reshape(batch * frames, channels, height, width),
            size=(self.image_size, self.image_size), mode="bicubic",
            align_corners=False, antialias=True,
        ).reshape(batch, frames, channels, self.image_size, self.image_size)
        video = (video - self.rgb_mean.float()) / self.rgb_std.float()
        return video.permute(0, 2, 1, 3, 4).contiguous()

    def encode_video(self, video):
        """No no_grad decorator: gradients must reach the perturbation generator."""
        features = self.vision_encoder(self.preprocess(video), use_image=False)
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, descriptions: Sequence[str], batch_size=4):
        if not hasattr(self, "text_encoder"):
            raise RuntimeError("Text encoder was released after feature caching")
        if not descriptions or any(not isinstance(x, str) or not x.strip() for x in descriptions):
            raise ValueError("Expected a nonempty sequence of action descriptions")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        features = []
        device = self.text_encoder.text_projection.device
        for begin in range(0, len(descriptions), batch_size):
            texts = ["summarize:" + x for x in descriptions[begin:begin + batch_size]]
            ids = self.tokenizer(
                texts, return_tensors="pt", max_length=80,
                truncation=True, padding="max_length",
            ).input_ids.to(device)
            encoded = self.text_encoder(ids)
            features.append(F.normalize(encoded.float(), dim=-1))
        return torch.cat(features)

    def release_text_encoder(self):
        """Release the 7B text tower after fixed class features are cached.

        The vision target and CLIP temperature stay resident and differentiable
        with respect to RGB input. This substantially lowers training memory.
        """
        if hasattr(self, "text_encoder"):
            del self.text_encoder
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self, video, text_features):
        video_features = self.encode_video(video)
        text_features = F.normalize(text_features.to(video_features).float(), dim=-1)
        return video_features @ text_features.T / self.temperature.float()


class InternVideo2Classifier(nn.Module):
    """Trainable supervised classifier initialized from InternVideo2-CLIP.

    The visual encoder is loaded with exactly the same Stage2 and CLIP overlay
    weights as :class:`InternVideo2CLIP`.  Keeping normalized visual features
    lets a text-feature initialized linear head reproduce the zero-shot logits
    before the first optimizer update.
    """

    def __init__(
        self,
        model_dir,
        num_classes: int,
        *,
        device="cuda",
        parameter_dtype=torch.float32,
        gradient_checkpointing=True,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        loaded = InternVideo2CLIP(
            model_dir,
            device=device,
            dtype=parameter_dtype,
            gradient_checkpointing=gradient_checkpointing,
            load_text_encoder=False,
        )
        vision_encoder = loaded.vision_encoder
        del loaded.vision_encoder
        self.vision_encoder = vision_encoder
        self.vision_encoder.requires_grad_(True)
        self.num_frames = loaded.num_frames
        self.image_size = loaded.image_size
        self.embedding_dim = loaded.embedding_dim
        self.num_classes = int(num_classes)
        self.load_report = loaded.load_report
        self.pretrained_temperature = float(loaded.temperature.float().item())
        self.register_buffer("rgb_mean", loaded.rgb_mean.detach().clone())
        self.register_buffer("rgb_std", loaded.rgb_std.detach().clone())
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes, bias=True)
        self.classifier.to(device=device, dtype=parameter_dtype)
        del loaded

    @torch.no_grad()
    def initialize_classifier(self, text_features: torch.Tensor) -> None:
        if text_features.shape != (self.num_classes, self.embedding_dim):
            raise ValueError(
                "Expected text features with shape "
                f"({self.num_classes}, {self.embedding_dim}), received {tuple(text_features.shape)}"
            )
        if not torch.isfinite(text_features).all():
            raise ValueError("Text features contain NaN/Inf")
        weights = F.normalize(text_features.float(), dim=-1) / self.pretrained_temperature
        self.classifier.weight.copy_(weights.to(self.classifier.weight))
        self.classifier.bias.zero_()

    def preprocess(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1:3] != (self.num_frames, 3):
            raise ValueError(
                f"Expected [B,{self.num_frames},3,H,W], received {tuple(video.shape)}"
            )
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        elif not video.is_floating_point():
            raise TypeError("Video must be uint8 or floating-point RGB in [0,1]")
        if not torch.isfinite(video).all().item():
            raise ValueError("Video contains NaN/Inf")
        lo, hi = video.detach().aminmax()
        if lo.item() < 0 or hi.item() > 1:
            raise ValueError("Float video must be in [0,1]; normalize only inside this wrapper")
        video = video.to(device=self.rgb_mean.device, dtype=torch.float32)
        batch, frames, channels, height, width = video.shape
        if (height, width) != (self.image_size, self.image_size):
            video = F.interpolate(
                video.reshape(batch * frames, channels, height, width),
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).reshape(batch, frames, channels, self.image_size, self.image_size)
        video = (video - self.rgb_mean.float()) / self.rgb_std.float()
        return video.permute(0, 2, 1, 3, 4).contiguous()

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(self.preprocess(video), use_image=False)
        return F.normalize(features.float(), dim=-1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.head_dropout(self.encode_video(video)))
