from torchvision.datasets import MNIST  # type: ignore[import]
from jax_dataloader.datasets import Dataset  # type: ignore[import]
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt


class MnistGridDataset(Dataset):
    """
    Dataset of images of 4x4 grids of MNIST hand-written digits.
    """

    def __init__(self, grid_wh: int = 4, num_classes: int = 10):
        self.grid_wh = grid_wh
        self.num_classes = num_classes

        def to_jnp(x):
            return jnp.array(x, dtype=jnp.float32)

        self.mnist_dataset = MNIST(
            root="./data",
            download=True,
            transform=to_jnp,
            train=True,
        )

        self.mnist_images = self.mnist_dataset.data
        self.mnist_labels = self.mnist_dataset.targets

        self.to_grid = to_grids_fn(self.grid_wh, self.num_classes)

    def __len__(self):
        return len(self.mnist_dataset) // (self.grid_wh**2)

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            items = [self[i.item()] for i in idx]
            original_grids = [original_grid for (original_grid, _) in items]
            colored_grids = [colored_grid for (_, colored_grid) in items]
            return (jnp.stack(original_grids), jnp.stack(colored_grids))

        idx_start = idx * (self.grid_wh**2)

        imgs = jnp.array(
            self.mnist_images[range(idx_start, idx_start + self.grid_wh**2)]
        )
        digits = jnp.array(
            self.mnist_labels[range(idx_start, idx_start + self.grid_wh**2)]
        )

        original_grid, colored_grid = self.to_grid(imgs, digits)

        return (original_grid, colored_grid)


def to_grids_fn(grid_wh: int, num_classes: int):

    @jax.jit
    def to_grids(imgs: jax.Array, digits: jax.Array):
        """
        Combine the original MNIST digit images into grids so they can be segmented.

        Returns a tuple of:
          * The original grayscale grid,
          * The true segmented grid with one-hot pixel channels.
        """
        # (16, 28, 28, 1) and (16, 28, 28, 11)
        digit_masks, one_hot_channeled_imgs = jax.vmap(to_mask_and_one_hot)(
            imgs,
            digits,
        )

        digit_grids = (
            digit_masks.reshape(grid_wh, grid_wh, 28, 28, 1)
            .transpose(0, 2, 1, 3, 4)
            .reshape(grid_wh * 28, grid_wh * 28, 1)
        )
        one_hot_channeled_grids = (
            one_hot_channeled_imgs.reshape(grid_wh, grid_wh, 28, 28, num_classes + 1)
            .transpose(0, 2, 1, 3, 4)
            .reshape(grid_wh * 28, grid_wh * 28, num_classes + 1)
        )

        # (112, 112, 1) and (112, 112, 11)
        return (digit_grids, one_hot_channeled_grids)

    def to_mask_and_one_hot(digit_mask: jax.Array, digit: jax.Array):
        # (W, H, 1C) grayscale img
        digit_mask = jnp.asarray(digit_mask)
        digit_mask = digit_mask / digit_mask.max()
        digit_mask = jnp.expand_dims(digit_mask, -1)

        # (W, H, 11C) one-hot channeled img
        one_hot_channeled_img = jnp.zeros_like(digit_mask).repeat(num_classes, -1)

        # add black background mask: all ones except where the digit is
        one_hot_channeled_img = jnp.concatenate(
            [one_hot_channeled_img, 1 - digit_mask], -1
        )

        # add digit mask to the correct one-hot channel
        one_hot_channeled_img = one_hot_channeled_img.at[:, :, digit].set(
            digit_mask.squeeze()
        )

        return digit_mask, one_hot_channeled_img


    return to_grids


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for idx in self.indices:
            yield self.dataset[idx]

    def __getitem__(self, idx):
        return self.dataset[idx]
