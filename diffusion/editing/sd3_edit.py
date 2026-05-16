from typing import List, Tuple, Optional

import torch
from tqdm import tqdm

from diffusion.base.sd3_sampler import SD3Euler, StableDiffusion3Base

# =======================================================================
# Factory
# =======================================================================

__EDITOR__ = {}

def register_editor(name:str):
    def wrapper(cls):
        if __EDITOR__.get(name, None) is not None:
            raise ValueError(f"Editor {name} already registered.")
        __EDITOR__[name] = cls
        return cls
    return wrapper

def get_editor(name:str, **kwargs):
    if name not in __EDITOR__:
        raise ValueError(f"Editor {name} does not exist.")
    return __EDITOR__[name](**kwargs)

# =======================================================================

@register_editor(name='dual')
class SD3Dual(SD3Euler):
    def sample(self, src_img:torch.Tensor, 
               src_prompt: str, tgt_prompt: str, null_prompt: str,
               NFE:int, img_shape:Optional[Tuple[int]],
               cfg_scale:float=7.0, n_start:int=10, batch_size:int=1,
               src_prompt_emb=None, tgt_prompt_emb=None, null_prompt_emb=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            src_prompt_emb = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_prompt_emb = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            null_prompt_emb = self.prepare_embed(null_prompt, null_prompt_emb)

        self.scheduler.set_timesteps(NFE, device=self.transformer.device, mu=self.shift)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        with torch.no_grad():
            xt = self.inversion(src_img, [src_prompt, null_prompt],
                                NFE, 
                                cfg_scale=0.0,
                                prompt_emb=src_prompt_emb,
                                null_emb=null_prompt_emb)

        pbar = tqdm(timesteps, total=NFE, desc='SD3 dual')
        for i, t in enumerate(pbar):
            timestep = t.expand(batch_size)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0
            
            with torch.no_grad():
                vxc = self.predict_vector(xt, timestep, tgt_prompt_emb[0], tgt_prompt_emb[1])
                vxn = self.predict_vector(xt, timestep, null_prompt_emb[0], null_prompt_emb[1])
                vx = vxn + cfg_scale * (vxc - vxn)

            xt = xt + (sigma_next-sigma) * vx

        # decode
        with torch.no_grad():
            img = self.decode(xt)
        return img 

@register_editor(name='sdedit')
class SD3SDEdit(SD3Euler):
    def sample(self, src_img:torch.Tensor, 
               src_prompt: str, tgt_prompt: str, null_prompt: str,
               NFE:int, img_shape:Optional[Tuple[int]],
               cfg_scale:float=7.0, n_start:int=10, batch_size:int=1,
               src_prompt_emb=None, tgt_prompt_emb=None, null_prompt_emb=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            src_prompt_emb = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_prompt_emb = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            null_prompt_emb = self.prepare_embed(null_prompt, null_prompt_emb)

        self.scheduler.set_timesteps(NFE, device=self.transformer.device, mu=self.shift)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        with torch.no_grad():
            zsrc = self.encode(src_img.to(self.vae.device).half())

        pbar = tqdm(timesteps, total=NFE, desc='SD3 SDEdit')
        for i, t in enumerate(pbar):
            timestep = t.expand(batch_size)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            if i < NFE - n_start:
                continue
            elif i == NFE-n_start:
                xt = (1-sigma) * zsrc + sigma * torch.randn_like(zsrc)
            
            with torch.no_grad():
                vxc = self.predict_vector(xt, timestep, tgt_prompt_emb[0], tgt_prompt_emb[1])
                vxn = self.predict_vector(xt, timestep, null_prompt_emb[0], null_prompt_emb[1])
                vx = vxn + cfg_scale * (vxc - vxn)

            xt = xt + (sigma_next-sigma) * vx

        # decode
        with torch.no_grad():
            img = self.decode(xt)
        return img 


@register_editor(name='flowedit')
class SD3EulerFE(SD3Euler):
    def sample(self, src_img:torch.Tensor, 
               src_prompt: str, tgt_prompt: str, null_prompt: str,
               NFE:int, img_shape:Optional[Tuple[int]],
               cfg_scale:float=13.5, n_start:int=33, batch_size:int=1,
               src_prompt_emb=None, tgt_prompt_emb=None, null_prompt_emb=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            src_prompt_emb = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_prompt_emb = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            null_prompt_emb = self.prepare_embed(null_prompt, null_prompt_emb)

        # initial
        with torch.no_grad():
            zsrc = self.encode(src_img.to(self.vae.device).half())
            zsrc = zsrc.to(self.transformer.device)
        z = zsrc.clone()

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.transformer.device, mu=self.shift)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3 FlowEdit')
        for i, t in enumerate(pbar):

            if i < NFE-n_start:  # skip
                continue

            timestep = t.expand(batch_size)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            noise = torch.randn_like(zsrc)
            ztsrc = (1-sigma) * zsrc + sigma * noise # forward
            zt = z + (ztsrc - zsrc)

            with torch.no_grad():
                # v for current estimate
                pred_v = self.predict_vector(zt, timestep, tgt_prompt_emb[0], tgt_prompt_emb[1])
                pred_vn = self.predict_vector(zt, timestep, null_prompt_emb[0], null_prompt_emb[1])
                pred_v = pred_vn + cfg_scale * (pred_v - pred_vn)
                
                # v for src estimate
                pred_vy = self.predict_vector(ztsrc, timestep, src_prompt_emb[0], src_prompt_emb[1])
                pred_vny = self.predict_vector(ztsrc, timestep, null_prompt_emb[0], null_prompt_emb[1])
                pred_vy = pred_vny + 3.5 * (pred_vy - pred_vny)

                dv = pred_v - pred_vy

            # next step
            z = z + (sigma_next - sigma) * dv

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img 
    
@register_editor(name='flowalign')
class SD3FlowAlign(SD3Euler):
    def sample(self, src_img:torch.Tensor, 
               src_prompt: str, tgt_prompt: str, null_prompt: str,
               NFE:int, img_shape:Optional[Tuple[int]],
               cfg_scale:float=13.5, n_start:int=17, batch_size:int=1,
               src_prompt_emb=None, tgt_prompt_emb=None, null_prompt_emb=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        if tgt_prompt_emb is None:
            with torch.no_grad():
                src_prompt_emb = self.prepare_embed(src_prompt, src_prompt_emb) 
                tgt_prompt_emb = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
                null_prompt_emb = self.prepare_embed(null_prompt, null_prompt_emb)

        # Follow the paper setup: use a 50-step SD3 schedule, then skip the
        # earliest noisy portion and keep the last 33 active steps.
        total_schedule_steps = int(NFE + max(0, n_start))
        self.scheduler.set_timesteps(total_schedule_steps, device=self.transformer.device, mu=self.shift)
        timesteps = self.scheduler.timesteps
        if n_start > 0:
            timesteps = timesteps[n_start:]
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        with torch.no_grad():
            zsrc = self.encode(src_img.to(self.vae.device).half())
        xt = zsrc.clone()

        pbar = tqdm(timesteps, total=len(timesteps), desc='SD3 FlowAlign')
        for i, t in enumerate(pbar):
            eps = torch.randn_like(zsrc)
            timestep = t.expand(batch_size)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < len(sigmas) else 0.0

            qt = (1-sigma)*zsrc + sigma*eps
            pt = xt + qt - zsrc

            with torch.no_grad():
                pt_batch = torch.cat([pt, pt], dim=0)
                prompt_batch = torch.cat([tgt_prompt_emb[0], src_prompt_emb[0]], dim=0)
                pooled_batch = torch.cat([tgt_prompt_emb[1], src_prompt_emb[1]], dim=0)
                vp_batch = self.predict_vectors(pt_batch, timestep.repeat(2), prompt_batch, pooled_batch)
                vpc, vpn = vp_batch.chunk(2, dim=0)
                vp = vpn + cfg_scale * (vpc - vpn)
                vqn = self.predict_vector(qt, timestep, src_prompt_emb[0], src_prompt_emb[1])
                vq = vqn

            xt = xt + (sigma_next-sigma) * (vp-vq) + 0.01 * (qt - sigma*vq - pt + sigma*vp)


        # decode
        with torch.no_grad():
            img = self.decode(xt)
        return img 


@register_editor(name='flowdinoalign')
class SD3FlowDinoAlign(SD3Euler):
    def __init__(
        self,
        model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
        device='cuda',
        shift: float = 3.0,
        dino_model: str = 'facebook/dinov2-small',
        dino_image_size: int = 128,
        dino_gamma: float = 0.01,
        dino_every_k: int = 4,
    ):
        super().__init__(model_key=model_key, device=device, shift=shift)
        self.dino_model = dino_model
        self.dino_image_size = dino_image_size
        self.dino_gamma = dino_gamma
        self.dino_every_k = max(1, int(dino_every_k))

    def sample(self, src_img:torch.Tensor,
               src_prompt: str, tgt_prompt: str, null_prompt: str,
               NFE:int, img_shape:Optional[Tuple[int]],
               cfg_scale:float=13.5, n_start:int=17, batch_size:int=1,
               src_prompt_emb=None, tgt_prompt_emb=None, null_prompt_emb=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        if tgt_prompt_emb is None:
            with torch.no_grad():
                src_prompt_emb = self.prepare_embed(src_prompt, src_prompt_emb)
                tgt_prompt_emb = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
                null_prompt_emb = self.prepare_embed(null_prompt, null_prompt_emb)

        total_schedule_steps = int(NFE + max(0, n_start))
        self.scheduler.set_timesteps(total_schedule_steps, device=self.transformer.device, mu=self.shift)
        timesteps = self.scheduler.timesteps
        if n_start > 0:
            timesteps = timesteps[n_start:]
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        with torch.no_grad():
            zsrc = self.encode(src_img.to(self.vae.device).half())
        xt = zsrc.clone()

        pbar = tqdm(timesteps, total=len(timesteps), desc='SD3 FlowDinoAlign')
        for i, t in enumerate(pbar):
            eps = torch.randn_like(zsrc)
            timestep = t.expand(batch_size)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < len(sigmas) else 0.0

            qt = (1 - sigma) * zsrc + sigma * eps
            pt = xt + qt - zsrc

            with torch.no_grad():
                pt_batch = torch.cat([pt, pt], dim=0)
                prompt_batch = torch.cat([tgt_prompt_emb[0], src_prompt_emb[0]], dim=0)
                pooled_batch = torch.cat([tgt_prompt_emb[1], src_prompt_emb[1]], dim=0)
                vp_batch = self.predict_vectors(pt_batch, timestep.repeat(2), prompt_batch, pooled_batch)
                vpc, vpn = vp_batch.chunk(2, dim=0)
                vp = vpn + cfg_scale * (vpc - vpn)
                vqn = self.predict_vector(qt, timestep, src_prompt_emb[0], src_prompt_emb[1])
                vq = vqn

            p0_hat = (pt - sigma * vp).detach()
            q0_hat = (qt - sigma * vq).detach()
            if i % self.dino_every_k == 0 or i == len(timesteps) - 1:
                v_modify = self.compute_dino_terminal_regularizer(
                    p0_hat,
                    q0_hat,
                    gamma=self.dino_gamma,
                    model_name=self.dino_model,
                    image_size=self.dino_image_size,
                )
            else:
                v_modify = torch.zeros_like(xt)

            xt = xt + (sigma_next - sigma) * (vp - vq) + v_modify

        with torch.no_grad():
            img = self.decode(xt)
        return img


@register_editor(name='flowlaplacianalign')
class SD3FlowLaplacianAlign(SD3Euler):
    def __init__(
        self,
        model_key: str = 'stabilityai/stable-diffusion-3-medium-diffusers',
        device='cuda',
        shift: float = 3.0,
        laplacian_image_size: int = 128,
        laplacian_levels: int = 2,
        laplacian_gamma: float = 0.01,
        laplacian_every_k: int = 4,
    ):
        super().__init__(model_key=model_key, device=device, shift=shift)
        self.laplacian_image_size = laplacian_image_size
        self.laplacian_levels = max(1, int(laplacian_levels))
        self.laplacian_gamma = laplacian_gamma
        self.laplacian_every_k = max(1, int(laplacian_every_k))

    def sample(self, src_img:torch.Tensor,
               src_prompt: str, tgt_prompt: str, null_prompt: str,
               NFE:int, img_shape:Optional[Tuple[int]],
               cfg_scale:float=13.5, n_start:int=17, batch_size:int=1,
               src_prompt_emb=None, tgt_prompt_emb=None, null_prompt_emb=None):

        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        if tgt_prompt_emb is None:
            with torch.no_grad():
                src_prompt_emb = self.prepare_embed(src_prompt, src_prompt_emb)
                tgt_prompt_emb = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
                null_prompt_emb = self.prepare_embed(null_prompt, null_prompt_emb)

        total_schedule_steps = int(NFE + max(0, n_start))
        self.scheduler.set_timesteps(total_schedule_steps, device=self.transformer.device, mu=self.shift)
        timesteps = self.scheduler.timesteps
        if n_start > 0:
            timesteps = timesteps[n_start:]
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        with torch.no_grad():
            zsrc = self.encode(src_img.to(self.vae.device).half())
        xt = zsrc.clone()

        pbar = tqdm(timesteps, total=len(timesteps), desc='SD3 FlowLaplacianAlign')
        for i, t in enumerate(pbar):
            eps = torch.randn_like(zsrc)
            timestep = t.expand(batch_size)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < len(sigmas) else 0.0

            qt = (1 - sigma) * zsrc + sigma * eps
            pt = xt + qt - zsrc

            with torch.no_grad():
                pt_batch = torch.cat([pt, pt], dim=0)
                prompt_batch = torch.cat([tgt_prompt_emb[0], src_prompt_emb[0]], dim=0)
                pooled_batch = torch.cat([tgt_prompt_emb[1], src_prompt_emb[1]], dim=0)
                vp_batch = self.predict_vectors(pt_batch, timestep.repeat(2), prompt_batch, pooled_batch)
                vpc, vpn = vp_batch.chunk(2, dim=0)
                vp = vpn + cfg_scale * (vpc - vpn)
                vqn = self.predict_vector(qt, timestep, src_prompt_emb[0], src_prompt_emb[1])
                vq = vqn

            p0_hat = (pt - sigma * vp).detach()
            q0_hat = (qt - sigma * vq).detach()
            if i % self.laplacian_every_k == 0 or i == len(timesteps) - 1:
                v_modify = self.compute_latent_laplacian_terminal_regularizer(
                    p0_hat,
                    q0_hat,
                    gamma=self.laplacian_gamma,
                    levels=self.laplacian_levels,
                )
            else:
                v_modify = torch.zeros_like(xt)

            xt = xt + (sigma_next - sigma) * (vp - vq) + v_modify

        with torch.no_grad():
            img = self.decode(xt)
        return img
