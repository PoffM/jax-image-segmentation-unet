import jax.numpy as jnp
import jax
import flax.linen as nn
from typing import Tuple, Generator


class DoubleConv(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jax.Array, train: bool):
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        return x


class UpsampleWithSkip(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jax.Array, skip: jax.Array, train: bool):
        x = nn.ConvTranspose(
            features=x.shape[-1] // 2, kernel_size=(2, 2), strides=(2, 2)
        )(x)
        x = jnp.concatenate([x, skip], axis=3)
        x = DoubleConv(self.out_channels)(x, train=train)
        return x


class SegmentationUNet(nn.Module):
    in_channels: int = 1
    num_classes: int = 10

    @nn.compact
    def __call__(self, x: jax.Array, train: bool):
        down_seq = [self.in_channels, 64, 128, 256]
        up_seq = [*down_seq[::-1][:-1], 64]

        skips = []
        for idx, (in_c, out_c) in twos(down_seq):
            x = DoubleConv(in_c)(x, train=train)
            skips.append(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        skips = skips[::-1]
        for idx, (in_c, out_c) in twos(up_seq):
            x = UpsampleWithSkip(out_c)(x, skips[idx], train=train)

        x = nn.Conv(features=self.num_classes + 1, kernel_size=(1, 1), strides=(1, 1))(
            x
        )

        x = nn.softmax(x, axis=-1)

        return x


def twos(nums: list[int]) -> Generator[Tuple[int, Tuple[int, int]], None, None]:
    for i in range(len(nums) - 1):
        yield i, (nums[i], nums[i + 1])
