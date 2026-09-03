from .sampler import Sampler
from .transport import Transport


def create_transport(config, time_dist_shift=1.0, time_dist_shift_eval=1.0):
    """Create a flow-matching Transport from a TransportConfig."""
    if getattr(config, "meanflow", None) is not None:
        raise ValueError("MeanFlow transport is not supported in MiniDog; set transport.meanflow to null.")
    return Transport(
        prediction=config.prediction,
        time_dist_type=config.time_dist_type,
        time_dist_shift=time_dist_shift,
        time_dist_shift_eval=time_dist_shift_eval,
        t_eps=config.t_eps,
        percep_loss_t_thresh=config.percep_loss_t_thresh,
    )


def create_sampler(transport, guidance_config):
    return Sampler(transport, guidance_config=guidance_config)


__all__ = ["create_transport", "create_sampler", "Transport", "Sampler"]
