

from __future__ import annotations
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .modules import MelSpec
from .model_utils import (
    default,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
)


class CFM(nn.Module):
   
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),


        audio_drop_prob=0.0,  
        cond_drop_prob=0.0,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    def build_sampling_sigmas(
        self,
        *,
        steps: int,
        dtype: torch.dtype,
        sigma_min: float | None = None,
        sigma_max: float = 1.0,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("`steps` must be positive.")

        if sigma_min is None:
            sigma_min = 1.0 / float(steps)

        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)

        # Avoid exact 0/1 singularities in DPM updates.
        eps = 1e-4
        sigma_min = max(sigma_min, eps)
        sigma_max = min(sigma_max, 1.0 - eps)

        if sigma_min >= sigma_max:
            raise ValueError(
                f"Expected sigma_min < sigma_max after clamping, got "
                f"sigma_min={sigma_min:.6f}, sigma_max={sigma_max:.6f}."
            )

        return torch.linspace(sigma_max, sigma_min, steps + 1, device=self.device, dtype=dtype)

    def build_sampling_time_grid(
        self,
        *,
        steps: int,
        dtype: torch.dtype,
        sigma_min: float | None = None,
        sigma_max: float = 1.0,
    ) -> torch.Tensor:
        sigmas = self.build_sampling_sigmas(
            steps=steps,
            dtype=dtype,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        return 1.0 - sigmas

    def prepare_condition(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
        elif cond.ndim == 3:
            if cond.shape[-1] == self.num_channels:
                cond = cond
            elif cond.shape[1] == self.num_channels:
                cond = cond.transpose(1, 2)
            else:
                raise ValueError(
                    f"Expected mel input with shape [B,N,D] or [B,D,N] and D={self.num_channels}, got {tuple(cond.shape)}"
                )
        else:
            raise ValueError(f"Unsupported condition shape: {tuple(cond.shape)}")

        cond = cond.to(device=self.device, dtype=next(self.parameters()).dtype).contiguous()
        if cond.shape[-1] != self.num_channels:
            raise ValueError(f"Condition mel channels must equal {self.num_channels}, got {cond.shape[-1]}")
        return cond

    def prepare_text(
        self,
        text: int["b nt"] | list[str] | list[list[str]] | None,  # noqa: F722
        batch: int,
        device: torch.device | None = None,
    ):
        if text is None:
            return None

        device = default(device, self.device)
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
        elif isinstance(text, torch.Tensor):
            text = text.to(device)
        else:
            raise TypeError(f"Unsupported text input type: {type(text)!r}")

        if text.shape[0] != batch:
            raise ValueError(f"Text batch size {text.shape[0]} does not match condition batch size {batch}")
        return text

    def predict_flow(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"] | float["b d n"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str] | list[list[str]] | None,  # noqa: F722
        time: float["b"] | float[""],  # noqa: F821 F722
        *,
        drop_audio_cond=False,
        drop_text=False,
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        x = self.prepare_condition(x)
        cond = self.prepare_condition(cond)
        text = self.prepare_text(text, batch=cond.shape[0], device=cond.device)
        return self.transformer(
            x=x,
            cond=cond,
            text=text,
            time=time,
            mask=mask,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )

    def compute_flow_target_loss(
        self,
        x_t: float["b n d"] | float["b d n"],  # noqa: F722
        flow_target: float["b n d"] | float["b d n"],  # noqa: F722
        cond: float["b n d"] | float["b d n"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str] | list[list[str]] | None,  # noqa: F722
        time: float["b"] | float[""],  # noqa: F821 F722
        *,
        weight: torch.Tensor | None = None,
        drop_audio_cond=False,
        drop_text=False,
        reduction: str = "mean",
    ):
        x_t = self.prepare_condition(x_t)
        flow_target = self.prepare_condition(flow_target)
        pred = self.predict_flow(
            x=x_t,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )
        loss = F.mse_loss(pred, flow_target, reduction="none")
        reduce_dims = tuple(range(1, loss.ndim))
        per_sample_loss = loss.mean(dim=reduce_dims)

        if weight is not None:
            weight = weight.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype).reshape(-1)
            if weight.shape[0] != per_sample_loss.shape[0]:
                raise ValueError("`weight` batch size must match loss batch size.")
            per_sample_loss = per_sample_loss * weight

        if reduction == "none":
            reduced_loss = per_sample_loss
        elif reduction == "mean":
            reduced_loss = per_sample_loss.mean()
        elif reduction == "sum":
            reduced_loss = per_sample_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        return reduced_loss, pred, per_sample_loss


    '''
        cond: noisy speech
        text: transcription
    '''

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
      
        *,
        steps=32,
        cfg_strength=1.0,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        drop_text=False,
        generator: torch.Generator | None = None,
        noise_init: torch.Tensor | None = None,
        solver: str = "dpm2",
        deterministic: bool = True,
        noise_level: float = 0.7,
        sigma_min: float | None = None,
        sigma_max: float = 1.0,
        deterministic_mask=None,
        sampling_time_grid=None,
        return_time_grid: bool = False,
    ):
        self.eval()
        cond = self.prepare_condition(cond)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        text = self.prepare_text(text, batch=batch, device=device)

        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)  # make sure lengths are at least those of the text characters
     
        mask = None

        if no_ref_audio:
            cond = torch.zeros_like(cond)
        step_cond = cond


        def fn(t, x):
    
            pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=False, drop_text=drop_text
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, time=t, mask=mask, drop_audio_cond=True, drop_text=True
            )
            return pred + (pred - null_pred) * cfg_strength

        if noise_init is not None and generator is not None:
            raise ValueError("Pass either `noise_init` or `generator`, not both.")

        if noise_init is not None:
            y0 = noise_init.to(device=cond.device, dtype=cond.dtype)
            if y0.shape != cond.shape:
                raise ValueError(f"`noise_init` shape {y0.shape} does not match condition shape {cond.shape}.")
        else:
            y0 = torch.randn(cond.shape, device=cond.device, dtype=cond.dtype, generator=generator)

        if solver not in {"flow", "dance", "ddim", "dpm1", "dpm2"}:
            raise ValueError(f"Unsupported solver: {solver!r}")

        try:
            from flow_nft.diffusers_patch.solver import run_sampling
        except Exception as exc:
            raise ImportError(
                "Speech discrete sampling requires `flow_nft.diffusers_patch.solver` "
                "and its dependencies."
            ) from exc

        if sampling_time_grid is None:
            sigmas = self.build_sampling_sigmas(
                steps=steps,
                dtype=step_cond.dtype,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            time_grid = 1.0 - sigmas
        else:
            if solver != "flow":
                raise ValueError("A custom speech sampling time grid currently supports only solver='flow'.")
            time_grid = torch.as_tensor(
                sampling_time_grid,
                device=step_cond.device,
                dtype=step_cond.dtype,
            ).reshape(-1)
            if int(time_grid.numel()) != int(steps) + 1:
                raise ValueError(
                    f"`sampling_time_grid` must contain steps + 1 = {int(steps) + 1} values, "
                    f"got {int(time_grid.numel())}."
                )
            if not bool(torch.isfinite(time_grid).all().item()):
                raise ValueError("`sampling_time_grid` must contain only finite values.")
            if float(time_grid[0].item()) < 0.0 or float(time_grid[-1].item()) > 1.0:
                raise ValueError("`sampling_time_grid` values must stay within [0, 1].")
            if not bool(torch.all(time_grid[1:] > time_grid[:-1]).item()):
                raise ValueError("`sampling_time_grid` must be strictly increasing.")
            sigmas = 1.0 - time_grid

        # Speech CFM predicts forward flow dX/dt. We pass -flow so that the
        # update direction remains consistent while sigma steps from high to low.
        def v_pred_fn(z, sigma):
            t = (1.0 - sigma).to(device=z.device, dtype=step_cond.dtype)
            return -fn(t, z)

        sampled, all_latents, _ = run_sampling(
            v_pred_fn=v_pred_fn,
            z=y0,
            sigma_schedule=sigmas,
            solver=solver,
            determistic=deterministic,
            eta=float(noise_level),
            deterministic_mask=deterministic_mask,
        )
        trajectory = torch.stack(all_latents, dim=0)
        out = sampled

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        if return_time_grid:
            return out, trajectory, time_grid
        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722 
        clean: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],  # noqa: F722
    ):
        '''
        inp: noisy speech
        clean: clean speech
        text: transcription
        
        '''
        
        inp = self.prepare_condition(inp)
        clean = self.prepare_condition(clean)

        batch, _, dtype, device, _ = *inp.shape[:2], inp.dtype, self.device, self.sigma
        text = self.prepare_text(text, batch=batch, device=device)

        # mel is x1
        x1 = clean

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        # cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        cond = inp
        
        
        # transformer and cfg training with a drop rate

        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        # flow matching loss
        loss, pred, _ = self.compute_flow_target_loss(
            x_t=φ,
            flow_target=flow,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            reduction="mean",
        )

        return loss, cond, pred


if __name__ == "__main__":
    model = CFM()
    
