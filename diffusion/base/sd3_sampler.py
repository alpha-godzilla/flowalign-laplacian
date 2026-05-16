from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

from tqdm import tqdm

try:
    import transformers
    if not hasattr(transformers, "CLIPTextModelWithProjection"):
        raise ImportError(
            "Your installed `transformers` package does not expose "
            "`CLIPTextModelWithProjection`, which Stable Diffusion 3 needs."
        )
    from diffusers import StableDiffusion3Pipeline
except Exception as e:
    raise RuntimeError(
        "Failed to import Stable Diffusion 3 dependencies. "
        "This project expects a newer Transformers/Diffusers stack, for example "
        "the versions pinned in `requirements.txt` (transformers==4.47.1, diffusers==0.33.1). "
        "Please reinstall the environment, then rerun the evaluation."
    ) from e

from diffusion import BaseSampler


_DINO_CACHE = {}


def _resolve_dino_model_name(model_name: str) -> str:
    """Map friendly aliases to actual Hugging Face model ids."""
    aliases = {
        "dinov2_vits14": "facebook/dinov2-small",
        "dinov2-vits14": "facebook/dinov2-small",
        "facebook/dinov2_vits14": "facebook/dinov2-small",
        "facebook/dinov2-vits14": "facebook/dinov2-small",
        "dinov2_vitb14": "facebook/dinov2-base",
        "dinov2-vitb14": "facebook/dinov2-base",
        "facebook/dinov2_vitb14": "facebook/dinov2-base",
        "facebook/dinov2-vitb14": "facebook/dinov2-base",
    }
    return aliases.get(model_name, model_name)

# =======================================================================
# Factory
# =======================================================================

__SAMPLER__ = {}

def register_sampler(name:str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None) is not None:
            raise ValueError(f"Sampler {name} already registered.")
        __SAMPLER__[name] = cls
        return cls
    return wrapper

def get_sampler(name:str, **kwargs):
    if name not in __SAMPLER__:
        raise ValueError(f"Sampler {name} does not exist.")
    return __SAMPLER__[name](**kwargs)

# =======================================================================


class StableDiffusion3Base(BaseSampler):
    def __init__(
        self,
        model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
        device='cuda',
        dtype=torch.float16,
        shift: float = 3.0,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.shift = shift

        # Support both directory-based and single-file loading.
        try:
            if model_key.endswith(".safetensors") or model_key.endswith(".ckpt"):
                pipe = StableDiffusion3Pipeline.from_single_file(
                    model_key, 
                    torch_dtype=self.dtype
                )
            else:
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    model_key, 
                    torch_dtype=self.dtype, 
                    local_files_only=True
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Stable Diffusion 3 pipeline from '{model_key}'. "
                "If using a directory, ensure it contains the full diffusers structure (model_index.json, etc.). "
                "If using a single file, ensure it is a valid .safetensors checkpoint. "
                f"Original error: {e}"
            )

        self.pipe = pipe
        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_3 = pipe.text_encoder_3

        self.vae = pipe.vae
        self.vae.eval()
        if hasattr(self.vae, "requires_grad_"):
            self.vae.requires_grad_(False)
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)-1) if hasattr(self, "vae") and self.vae is not None else 8
        )

    def _set_timesteps(self, num_inference_steps: int, device: torch.device, n_start: int = 0):
        """Configure SD3/FlowMatch timesteps with optional early-step skipping.

        For PIE-Bench reproduction we want to preserve the SD3 time shift and, for
        FlowAlign, emulate the FlowEdit-style protocol of skipping the early noisy
        portion while keeping the last active steps.
        """
        total_steps = int(num_inference_steps) + int(max(0, n_start))
        try:
            self.scheduler.set_timesteps(total_steps, device=device, mu=self.shift)
        except TypeError:
            # Fallback for older/newer diffusers signatures.
            self.scheduler.set_timesteps(total_steps, device=device)

        timesteps = self.scheduler.timesteps
        if n_start > 0:
            timesteps = timesteps[n_start:]

        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        return timesteps, sigmas

    def encode_prompt(self, prompt: List[str], batch_size:int=1) -> List[torch.Tensor]:
        '''
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        '''
        # Use the pipeline's built-in encode_prompt for better compatibility with missing components (like T5)
        prompt_emb, _, pooled_prompt_emb, _ = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=None,
            device=self.transformer.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return prompt_emb, pooled_prompt_emb


    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):
        H, W = img_size
        lH, lW = H//self.vae_scale_factor, W//self.vae_scale_factor
        lC = self.transformer.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z-self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z/self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        v = self.transformer(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False)[0]
        return v

    def predict_vectors(self, z, t, prompt_emb, pooled_emb):
        """Batched version of predict_vector for multiple prompt conditions."""
        return self.transformer(
            hidden_states=z,
            timestep=t,
            pooled_projections=pooled_emb,
            encoder_hidden_states=prompt_emb,
            return_dict=False,
        )[0]

    def load_dino_backbone(
        self,
        model_name: str = "facebook/dinov2-small",
    ):
        """Load and cache a frozen DINOv2 backbone."""
        model_name = _resolve_dino_model_name(model_name)
        cache_key = (model_name, str(self.device))
        if cache_key in _DINO_CACHE:
            return _DINO_CACHE[cache_key]

        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        model.eval()
        model.requires_grad_(False)
        _DINO_CACHE[cache_key] = (processor, model)
        return processor, model

    def normalize_for_dino(self, image: torch.Tensor, image_size: int = 224) -> torch.Tensor:
        """Resize and ImageNet-normalize an image tensor for DINO."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image tensor for DINO, got {tuple(image.shape)}")

        image = image.float()
        if image.min().item() < 0:
            image = (image + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        image = (image - mean) / std
        return image

    def _normalize_for_pyramid(self, image: torch.Tensor, image_size: int = 256) -> torch.Tensor:
        """Resize an image tensor to a compact working size for pyramid losses."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image tensor for pyramid loss, got {tuple(image.shape)}")

        image = image.float()
        if image.min().item() < 0:
            image = (image + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return image

    @staticmethod
    def _gaussian_blur(image: torch.Tensor) -> torch.Tensor:
        """Apply a fixed 5x5 Gaussian blur depthwise."""
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image tensor for Gaussian blur, got {tuple(image.shape)}")
        channels = image.shape[1]
        kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=image.device, dtype=image.dtype)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
        image = F.pad(image, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(image, kernel, groups=channels)

    def _laplacian_pyramid(self, image: torch.Tensor, levels: int = 3) -> List[torch.Tensor]:
        """Build a differentiable Laplacian pyramid for [B, C, H, W] tensors."""
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image tensor for pyramid loss, got {tuple(image.shape)}")
        levels = max(1, int(levels))
        current = image
        pyramid: List[torch.Tensor] = []
        for _ in range(levels):
            blurred = self._gaussian_blur(current)
            down = F.avg_pool2d(blurred, kernel_size=2, stride=2)
            up = F.interpolate(down, size=current.shape[-2:], mode="bilinear", align_corners=False)
            pyramid.append(current - up)
            current = down
        pyramid.append(current)
        return pyramid

    def _latent_pyramid(self, latent: torch.Tensor, levels: int = 3) -> List[torch.Tensor]:
        """Build a lightweight Laplacian-style pyramid directly in latent space."""
        if latent.ndim != 4:
            raise ValueError(f"Expected 4D latent tensor for pyramid loss, got {tuple(latent.shape)}")
        levels = max(1, int(levels))
        current = latent
        pyramid: List[torch.Tensor] = []
        for _ in range(levels):
            blurred = self._gaussian_blur(current)
            down = F.avg_pool2d(blurred, kernel_size=2, stride=2)
            up = F.interpolate(down, size=current.shape[-2:], mode="bilinear", align_corners=False)
            pyramid.append(current - up)
            current = down
        pyramid.append(current)
        return pyramid

    def compute_laplacian_terminal_regularizer(
        self,
        p0_latent: torch.Tensor,
        q0_latent: torch.Tensor,
        gamma: float = 0.01,
        image_size: int = 256,
        levels: int = 3,
    ) -> torch.Tensor:
        """Compute -gamma * d/dp0 0.5 * sum_k ||L_k(p0) - L_k(q0)||^2."""
        p0_latent = p0_latent.detach().requires_grad_(True)
        q0_latent = q0_latent.detach()

        with torch.enable_grad():
            latents = torch.cat([p0_latent, q0_latent], dim=0)
            imgs = self.decode(latents)
            p0_img, q0_img = imgs.chunk(2, dim=0)
            p0_in = self._normalize_for_pyramid(p0_img, image_size=image_size)
            q0_in = self._normalize_for_pyramid(q0_img, image_size=image_size)

            p_pyr = self._laplacian_pyramid(p0_in, levels=levels)
            q_pyr = self._laplacian_pyramid(q0_in, levels=levels)
            loss = 0.0
            for p_feat, q_feat in zip(p_pyr, q_pyr):
                diff = p_feat - q_feat
                loss = loss + 0.5 * diff.flatten(1).pow(2).mean(dim=1).mean()
            grad_p0 = torch.autograd.grad(loss, p0_latent, retain_graph=False, create_graph=False)[0]

        return -gamma * grad_p0

    def compute_latent_laplacian_terminal_regularizer(
        self,
        p0_latent: torch.Tensor,
        q0_latent: torch.Tensor,
        gamma: float = 0.01,
        levels: int = 2,
    ) -> torch.Tensor:
        """Compute -gamma * d/dp0 0.5 * sum_k ||L_k(p0) - L_k(q0)||^2 in latent space."""
        p0_latent = p0_latent.detach().requires_grad_(True)
        q0_latent = q0_latent.detach()

        with torch.enable_grad():
            p_pyr = self._latent_pyramid(p0_latent, levels=levels)
            q_pyr = self._latent_pyramid(q0_latent, levels=levels)
            loss = 0.0
            for p_feat, q_feat in zip(p_pyr, q_pyr):
                diff = p_feat - q_feat
                loss = loss + 0.5 * diff.flatten(1).pow(2).mean(dim=1).mean()
            grad_p0 = torch.autograd.grad(loss, p0_latent, retain_graph=False, create_graph=False)[0]

        return -gamma * grad_p0

    def compute_dino_terminal_regularizer(
        self,
        p0_latent: torch.Tensor,
        q0_latent: torch.Tensor,
        gamma: float = 0.01,
        model_name: str = "facebook/dinov2-small",
        image_size: int = 224,
    ) -> torch.Tensor:
        """Compute -gamma * d/dp0 0.5||f(p0)-f(q0)||^2, where f is frozen DINO."""
        _, dino_model = self.load_dino_backbone(model_name)

        p0_latent = p0_latent.detach().requires_grad_(True)
        q0_latent = q0_latent.detach()

        with torch.enable_grad():
            latents = torch.cat([p0_latent, q0_latent], dim=0)
            imgs = self.decode(latents)
            p0_img, q0_img = imgs.chunk(2, dim=0)

            dino_in = torch.cat(
                [
                    self.normalize_for_dino(p0_img, image_size=image_size),
                    self.normalize_for_dino(q0_img, image_size=image_size),
                ],
                dim=0,
            )

            feats = dino_model(pixel_values=dino_in).last_hidden_state.mean(dim=1)
            fp, fq = feats.chunk(2, dim=0)
            diff = fp - fq
            loss = 0.5 * diff.flatten(1).pow(2).sum(dim=1).mean()
            grad_p0 = torch.autograd.grad(loss, p0_latent, retain_graph=False, create_graph=False)[0]

        return -gamma * grad_p0

    def prepare_embed(self, prompt:str, embs: List[torch.Tensor]) -> List[torch.Tensor]:
        '''
        Return prompt embedding and pooled embedding.
        '''
        if embs is None:
            prompt_emb, pooled_emb = self.encode_prompt(prompt)
        else:
            prompt_emb, pooled_emb = embs
        
        prompt_emb = prompt_emb.to(self.transformer.device)
        pooled_emb = pooled_emb.to(self.transformer.device)
        return prompt_emb, pooled_emb

@register_sampler('euler')
class SD3Euler(StableDiffusion3Base):
    def __init__(
        self,
        model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
        device='cuda',
        shift: float = 3.0,
    ):
        super().__init__(model_key=model_key, device=device, shift=shift)

    def inversion(self, src_img, prompts: List[str], NFE:int, cfg_scale: float=1.0, batch_size: int=1,
                  prompt_emb:Optional[List[torch.Tensor]]=None,
                  null_emb:Optional[List[torch.Tensor]]=None):

        # encode text prompts
        with torch.no_grad():
            prompt_emb, pooled_emb = self.prepare_embed(prompts[0], prompt_emb)
            null_prompt_emb, null_pooled_emb = self.prepare_embed(prompts[1], null_emb)

        # initialize latent
        src_img = src_img.to(device=self.vae.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img).to(self.transformer.device)

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.transformer.device, mu=self.shift)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.transformer.device)])
        timesteps = torch.flip(timesteps, dims=[0])
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps[:-1], total=NFE, desc='SD3 Euler Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.transformer.device)
            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1]

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        return z

    def sample(self, prompts: List[str], NFE:int, img_shape: Optional[Tuple[int]]=None,
               cfg_scale: float=1.0, batch_size: int = 1,
               latent:Optional[List[torch.Tensor]]=None,
               prompt_emb:Optional[List[torch.Tensor]]=None,
               null_emb:Optional[List[torch.Tensor]]=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            prompt_emb, pooled_emb = self.prepare_embed(prompts[0], prompt_emb)
            null_prompt_emb, null_pooled_emb = self.prepare_embed(prompts[1], null_emb)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.device, mu=self.shift)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3 Euler')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)

            with torch.no_grad():
                pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
                if cfg_scale != 1.0:
                    pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
                else:
                    pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img
