import torch

from qat.eval.vllm_compat import reshape_channel_scales


def test_reshape_channel_scales_accepts_torch_size_and_keeps_batch_dims() -> None:
    scales = torch.arange(12).reshape(6, 2)
    reshaped = reshape_channel_scales(scales, torch.Size([3, 4]))

    assert reshaped.shape == (3, 4)
    torch.testing.assert_close(reshaped, scales.reshape(3, 4))
