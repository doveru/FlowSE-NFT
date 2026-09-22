import math
import torch
from typing import Optional, List
from dataclasses import dataclass
import torch.distributed as dist
from tqdm.auto import tqdm

try:
    from diffusers.utils.torch_utils import randn_tensor as _diffusers_randn_tensor
except Exception:
    _diffusers_randn_tensor = None


def randn_tensor(shape, generator=None, device=None, dtype=None):
    if _diffusers_randn_tensor is not None:
        return _diffusers_randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype

    if isinstance(generator, list):
        if len(generator) == 0:
            raise ValueError("`generator` list cannot be empty.")
        if len(shape) == 0 or shape[0] != len(generator):
            raise ValueError(
                "When `generator` is a list, the first dimension of `shape` must match list length."
            )
        per_sample = (1, *shape[1:])
        parts = [torch.randn(per_sample, generator=g, **kwargs) for g in generator]
        return torch.cat(parts, dim=0)

    if generator is not None:
        kwargs["generator"] = generator
    return torch.randn(shape, **kwargs)

def _resolve_deterministic_mask(deterministic_mask, num_steps: int, default: bool) -> list[bool]:
    if num_steps <= 0:
        raise ValueError("`num_steps` must be positive.")
    if deterministic_mask is None:
        return [bool(default)] * num_steps

    mask = torch.as_tensor(deterministic_mask, dtype=torch.bool).reshape(-1)
    if int(mask.numel()) != int(num_steps):
        raise ValueError(
            f"`deterministic_mask` must have length {num_steps}, got {int(mask.numel())}."
        )
    return [bool(value) for value in mask.tolist()]


def _safe_normal_log_prob(prev_sample, prev_sample_mean, scale):
    scale = torch.as_tensor(scale, device=prev_sample.device, dtype=prev_sample.dtype)
    if torch.any(scale <= 0):
        return torch.zeros(prev_sample.shape[0], device=prev_sample.device, dtype=prev_sample.dtype)

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (scale**2))
        - torch.log(scale)
        - torch.log(torch.sqrt(torch.as_tensor(2 * math.pi, device=prev_sample.device, dtype=prev_sample.dtype)))
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


# Modified from MixGRPO
def run_sampling(
    v_pred_fn,
    z,
    sigma_schedule,
    solver="flow",
    determistic=False,
    eta=0.7,
    deterministic_mask=None,
):
    assert solver in ["flow", "dance", "ddim", "dpm1", "dpm2"]
    dtype = z.dtype
    num_steps = len(sigma_schedule) - 1
    per_step_deterministic = _resolve_deterministic_mask(
        deterministic_mask,
        num_steps,
        determistic,
    )
    if deterministic_mask is not None and solver != "flow" and len(set(per_step_deterministic)) > 1:
        raise ValueError("Mixed SDE/ODE timestep masks are only supported for solver='flow'.")

    all_latents = [z]
    all_log_probs = []

    if "dpm" in solver:
        order = int(solver[-1])
        dpm_state = DPMState(order=order)
    for i in tqdm(
        range(num_steps),
        desc="Sampling Progress",
        dynamic_ncols=True,
        disable=not dist.is_initialized() or dist.get_rank() != 0,
    ):
        sigma = sigma_schedule[i]
        step_deterministic = per_step_deterministic[i]

        pred = v_pred_fn(z.to(dtype), sigma)
        if solver == "flow":
            z, pred_original, log_prob = flow_nft_step(
                model_output=pred.float(),
                latents=z.float(),
                eta=eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
                determistic=step_deterministic,
            )
        elif solver == "dance":
            z, pred_original, log_prob = dance_grpo_step(
                pred.float(),
                z.float(),
                eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
                determistic=step_deterministic,
            )
        elif solver == "ddim":
            z, pred_original, log_prob = ddim_step(
                pred.float(),
                z.float(),
                eta,
                sigmas=sigma_schedule,
                index=i,
                prev_sample=None,
                determistic=step_deterministic,
            )
        elif "dpm" in solver:
            if not step_deterministic:
                raise ValueError("DPM solvers only support deterministic ODE sampling.")
            z, pred_original, log_prob = dpm_step(
                order,
                model_output=pred.float(),
                sample=z.float(),
                step_index=i,
                timesteps=sigma_schedule[:-1],
                sigmas=sigma_schedule,
                dpm_state=dpm_state,
            )
        else:
            assert False
        z = z.to(dtype)
        all_latents.append(z)
        all_log_probs.append(log_prob)

    latents = z.to(dtype)
    # all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
    # all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
    return latents, all_latents, all_log_probs


def flow_nft_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
):
    device = model_output.device
    sigma = sigmas[index].to(device)
    sigma_prev = sigmas[index + 1].to(device)
    # Use the second sigma as a safe proxy for the "max usable sigma".
    # This prevents the first near-1 step from exploding std when schedules
    # start at 1.0 or are clipped very close to 1.0 (e.g. 0.9999).
    sigma_max = sigmas[1].to(device=device, dtype=sigma.dtype)
    dt = sigma_prev - sigma  # neg dt

    pred_original_sample = latents - sigma * model_output

    sigma_ref = torch.where(sigma >= sigma_max, sigma_max, sigma)
    denom = torch.clamp(1.0 - sigma_ref, min=1e-4)
    num = torch.clamp(sigma, min=1e-8)
    std_dev_t = torch.sqrt(num / denom) * eta

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    prev_sample_mean = (
        latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is None:
        variance_noise = randn_tensor(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    if determistic:
        prev_sample = latents + dt * model_output

    log_prob = _safe_normal_log_prob(
        prev_sample,
        prev_sample_mean,
        std_dev_t * torch.sqrt(-1 * dt),
    )

    return prev_sample, pred_original_sample, log_prob


def dance_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    determistic: bool = False,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma  # neg dt
    ode_prev_sample = latents + dsigma * model_output
    prev_sample_mean = ode_prev_sample

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]  # pos -dt
    std_dev_t = eta * math.sqrt(delta_t)

    score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
    log_term = -0.5 * eta**2 * score_estimate
    prev_sample_mean = prev_sample_mean + log_term * dsigma

    if prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if determistic:
        prev_sample = ode_prev_sample

    log_prob = _safe_normal_log_prob(prev_sample, prev_sample_mean, std_dev_t)
    return prev_sample, pred_original_sample, log_prob


def ddim_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    determistic: bool = False,
):
    model_output = convert_model_output(model_output, latents, sigmas, step_index=index)
    prev_sample, prev_sample_mean, std_dev_t, dt_sqrt = ddim_update(
        model_output,
        sigmas.to(torch.float64),
        index,
        latents,
        eta=eta,
    )

    if determistic:
        prev_sample, _, _, _ = ddim_update(
            model_output,
            sigmas.to(torch.float64),
            index,
            latents,
            eta=0.0,
        )

    log_prob = _safe_normal_log_prob(prev_sample, prev_sample_mean, std_dev_t * dt_sqrt)
    return prev_sample, model_output, log_prob


@dataclass
class DPMState:
    order: int
    model_outputs: List[torch.Tensor] = None
    lower_order_nums = 0

    def __post_init__(self):
        self.model_outputs = [None] * self.order

    def update(self, model_output: torch.Tensor):
        for i in range(self.order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

    def update_lower_order(self):
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1


def dpm_step(
    order,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    step_index: int,
    timesteps: list,
    sigmas: torch.Tensor,
    dpm_state: DPMState = None,
) -> torch.Tensor:

    # Improve numerical stability for small number of steps
    lower_order_final = step_index == len(timesteps) - 1
    lower_order_second = (step_index == len(timesteps) - 2) and len(timesteps) < 15

    model_output = convert_model_output(model_output, sample, sigmas, step_index=step_index)

    assert dpm_state is not None
    dpm_state.update(model_output)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    if order == 1 or dpm_state.lower_order_nums < 1 or lower_order_final:
        if step_index == 0 or lower_order_final:
            prev_sample, _, _, _ = ddim_update(
                model_output,
                sigmas.to(torch.float64),
                step_index,
                sample,
                eta=0.0,
            )
        else:
            prev_sample = dpm_solver_first_order_update(
                model_output,
                sigmas.to(torch.float64),
                step_index,
                sample,
            )
    elif order == 2 or dpm_state.lower_order_nums < 2 or lower_order_second:
        prev_sample = multistep_dpm_solver_second_order_update(
            dpm_state.model_outputs,
            sigmas.to(torch.float64),
            step_index,
            sample,
        )
    else:
        assert False

    dpm_state.update_lower_order()

    # Cast sample back to expected dtype
    prev_sample = prev_sample.to(model_output.dtype)

    return prev_sample, model_output, None


def convert_model_output(
    model_output,
    sample,
    sigmas,
    step_index,
) -> torch.Tensor:
    sigma_t = sigmas[step_index]
    x0_pred = sample - sigma_t * model_output

    return x0_pred


def ddim_update(
    model_output: torch.Tensor,
    sigmas,
    step_index,
    sample: torch.Tensor = None,
    noise: Optional[torch.Tensor] = None,
    eta: float = 1.0,
) -> torch.Tensor:

    t, s = sigmas[step_index + 1], sigmas[step_index]

    std_dev_t = eta * t
    dt_sqrt = torch.sqrt(1.0 - t**2 * (1 - s) ** 2 / (s**2 * (1 - t) ** 2))
    rho_t = std_dev_t * dt_sqrt
    noise_pred = (sample - (1 - s) * model_output) / s
    if noise is None:
        noise = torch.randn_like(model_output)
    prev_mean = (1 - t) * model_output + torch.sqrt(t**2 - rho_t**2) * noise_pred
    x_t = prev_mean + rho_t * noise

    return x_t, prev_mean, std_dev_t, dt_sqrt


def dpm_solver_first_order_update(
    model_output: torch.Tensor,
    sigmas,
    step_index,
    sample: torch.Tensor = None,
) -> torch.Tensor:

    sigma_t, sigma_s = sigmas[step_index + 1], sigmas[step_index]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = _sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s
    x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output

    return x_t


def multistep_dpm_solver_second_order_update(
    model_output_list: List[torch.Tensor],
    sigmas,
    step_index,
    sample: torch.Tensor = None,
) -> torch.Tensor:

    sigma_t, sigma_s0, sigma_s1 = (
        sigmas[step_index + 1],
        sigmas[step_index],
        sigmas[step_index - 1],
    )

    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = _sigma_to_alpha_sigma_t(sigma_s1)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

    m0, m1 = model_output_list[-1], model_output_list[-2]

    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)

    x_t = (
        (sigma_t / sigma_s0) * sample
        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
    )

    return x_t


def _sigma_to_alpha_sigma_t(sigma):
    alpha_t = 1 - sigma
    sigma_t = sigma
    return alpha_t, sigma_t
