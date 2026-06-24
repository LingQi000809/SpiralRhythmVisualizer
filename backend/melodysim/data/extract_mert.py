import torch

def run_mert_model_and_get_features(waveforms, audio_model, time_reduce=None):
    """Extract MERT features from waveforms.

    Args:
        waveforms: torch tensor (batch, num_samples)
        audio_model: AutoModel from "m-a-p/MERT-v1-95M"
        time_reduce: AvgPool1d(kernel_size=10, stride=10) — created if None
    Returns:
        Tensor of shape (batch, 4, reduced_frames, 768)
    """
    if time_reduce is None:
        time_reduce = torch.nn.AvgPool1d(kernel_size=10, stride=10, count_include_pad=False)
    hidden_states = audio_model(waveforms, output_hidden_states=True).hidden_states
    # Take every 3rd hidden state starting from index 2 (layers 2, 5, 8, 11) → 4 layers
    audio_features = torch.stack(
        [time_reduce(h.detach()[:, :, :].permute(0, 2, 1)).permute(0, 2, 1) for h in hidden_states[2::3]],
        dim=1
    )
    return audio_features
