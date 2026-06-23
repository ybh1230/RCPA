from .base_pipeline import BasePipeline
import torch
import torch.nn.functional as F

def _sample_training_timestep(pipe: BasePipeline, inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    if max_timestep_boundary <= min_timestep_boundary:
        raise ValueError(
            f"Invalid timestep boundary: min={min_timestep_boundary}, max={max_timestep_boundary}."
        )
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    return timestep_id, timestep


def _scheduler_sigma(pipe: BasePipeline, timestep_id, target):
    sigma = pipe.scheduler.sigmas[timestep_id].to(device=target.device, dtype=target.dtype)
    return sigma.view(1, *([1] * (target.ndim - 1)))


def _model_prediction(pipe: BasePipeline, inputs, timestep):
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    return pipe.model_fn(**models, **inputs, timestep=timestep)


def _parse_float_list(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
        return parsed if parsed else default
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def _degrade_latents_down_up(x0, factor=2.0, shift=(0, 0)):
    factor = max(float(factor), 1.0)
    if factor == 1.0:
        return x0
    _, _, t, h, w = x0.shape
    down_h = max(1, int(round(h / factor)))
    down_w = max(1, int(round(w / factor)))
    shifted = torch.roll(x0, shifts=shift, dims=(-2, -1)) if shift != (0, 0) else x0
    degraded = F.interpolate(
        shifted,
        size=(t, down_h, down_w),
        mode="trilinear",
        align_corners=False,
    )
    restored = F.interpolate(
        degraded,
        size=(t, h, w),
        mode="trilinear",
        align_corners=False,
    )
    return torch.roll(restored, shifts=(-shift[0], -shift[1]), dims=(-2, -1)) if shift != (0, 0) else restored


def _spatial_high_pass(x):
    x_float = x.float()
    low_pass = F.avg_pool3d(
        x_float,
        kernel_size=(1, 3, 3),
        stride=1,
        padding=(0, 1, 1),
    )
    return x_float - low_pass


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchDownUpReconstructionLoss(pipe: BasePipeline, **inputs):
    timestep_id, timestep = _sample_training_timestep(pipe, inputs)
    x0 = inputs["input_latents"]
    factor = float(inputs.get("downsample_factor", 2.0))
    x0_degraded = _degrade_latents_down_up(x0, factor=factor)

    noise = torch.randn_like(x0)
    x_noisy = pipe.scheduler.add_noise(x0_degraded, noise, timestep)
    model_inputs = dict(inputs)
    model_inputs["latents"] = x_noisy

    flow_pred = _model_prediction(pipe, model_inputs, timestep)
    sigma = _scheduler_sigma(pipe, timestep_id, x_noisy)
    x0_pred = x_noisy - sigma * flow_pred

    loss = torch.nn.functional.mse_loss(x0_pred.float(), x0.float())
    return loss * pipe.scheduler.training_weight(timestep)


def FlowMatchMultiViewAgreementLoss(pipe: BasePipeline, **inputs):
    timestep_id, timestep = _sample_training_timestep(pipe, inputs)
    x0 = inputs["input_latents"]
    num_views = max(1, int(inputs.get("mvla_num_views", 3)))
    down_factors = _parse_float_list(inputs.get("mvla_down_factors", "2,4"), [2.0, 4.0])
    reconstruction_weight = float(inputs.get("mvla_reconstruction_weight", 1.0))
    consistency_weight = float(inputs.get("mvla_agreement_weight", 0.15))
    high_freq_weight = float(inputs.get("mvla_high_freq_weight", 0.25))

    noise = torch.randn_like(x0)
    sigma = _scheduler_sigma(pipe, timestep_id, x0)
    predictions = []
    reconstruction_loss = x0.float().new_zeros(())
    high_frequency_loss = x0.float().new_zeros(())
    target_high = _spatial_high_pass(x0) if high_freq_weight > 0 else None

    for view_id in range(num_views):
        factor = down_factors[view_id % len(down_factors)]
        phase = max(1, int(round(factor)))
        shift = (view_id % phase, (view_id * 2) % phase)
        x0_degraded = _degrade_latents_down_up(x0, factor=factor, shift=shift)
        x_noisy = pipe.scheduler.add_noise(x0_degraded, noise, timestep)

        model_inputs = dict(inputs)
        model_inputs["latents"] = x_noisy
        flow_pred = _model_prediction(pipe, model_inputs, timestep)
        x0_pred = x_noisy - sigma * flow_pred
        predictions.append(x0_pred)

        reconstruction_loss = reconstruction_loss + torch.nn.functional.mse_loss(
            x0_pred.float(),
            x0.float(),
        )
        if high_freq_weight > 0:
            high_frequency_loss = high_frequency_loss + torch.nn.functional.mse_loss(
                _spatial_high_pass(x0_pred),
                target_high,
            )

    prediction_stack = torch.stack(predictions, dim=0)
    consensus = prediction_stack.mean(dim=0, keepdim=True).detach()
    consistency_loss = ((prediction_stack.float() - consensus.float()) ** 2).mean()
    reconstruction_loss = reconstruction_loss / num_views
    high_frequency_loss = high_frequency_loss / num_views

    loss = (
        reconstruction_weight * reconstruction_loss
        + consistency_weight * consistency_loss
        + high_freq_weight * high_frequency_loss
    )
    return loss * pipe.scheduler.training_weight(timestep)


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
