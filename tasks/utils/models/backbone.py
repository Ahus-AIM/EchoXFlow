from __future__ import annotations

from torch import Tensor, nn

Int3 = tuple[int, int, int]


class DenseVideoBackbone(nn.Module):
    """Common contract for dense frame/video backbones used by benchmark task heads."""

    out_channels: int
    strides: tuple[Int3, ...]
    output_spatial_stride: tuple[int, int]

    def forward(self, video: Tensor) -> Tensor:
        """Map ``[B,C,T,H,W]`` video to ``[B,F,T,H',W']`` dense features."""
        raise NotImplementedError


class FramewiseDenseVideoBackbone(DenseVideoBackbone):
    """Apply one shared 2D dense model independently to every video frame."""

    def forward_frames(self, frames: Tensor) -> Tensor:
        """Map ``[N,C,H,W]`` frames to ``[N,F,H',W']`` dense features."""
        raise NotImplementedError

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,C,T,H,W], got {tuple(video.shape)}")
        batch_size, channels, time_steps, height, width = (int(value) for value in video.shape)
        frame_batch = video.permute(0, 2, 1, 3, 4).reshape(batch_size * time_steps, channels, height, width)
        prediction = self.forward_frames(frame_batch)
        if prediction.ndim != 4:
            raise ValueError(f"framewise backbone must return [N,F,H,W], got {tuple(prediction.shape)}")
        output_channels, output_height, output_width = (int(value) for value in prediction.shape[1:])
        if output_channels != int(self.out_channels):
            raise ValueError(
                f"framewise backbone declared {self.out_channels} output channels but returned {output_channels}"
            )
        return (
            prediction.reshape(batch_size, time_steps, output_channels, output_height, output_width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )


__all__ = ["DenseVideoBackbone", "FramewiseDenseVideoBackbone", "Int3"]
