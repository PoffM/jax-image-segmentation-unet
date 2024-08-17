import jax_dataloader as jdl  # type: ignore
from mnist_grid_dataset import MnistGridDataset, SubsetDataset


def init_loaders(grid_wh: int, batch_size: int):
    dataset = MnistGridDataset(grid_wh=grid_wh, num_classes=10)

    idxs = range(len(dataset))
    split_point = int(0.8 * len(dataset))

    train_set = SubsetDataset(dataset, idxs[:split_point])
    val_set = SubsetDataset(dataset, idxs[split_point:])

    train_loader = jdl.DataLoader(
        train_set,  # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
        backend="jax",  # Use 'jax' backend for loading data
        batch_size=batch_size,  # Batch size
        shuffle=True,  # Shuffle the dataloader every iteration or not
        drop_last=False,  # Drop the last batch or not
    )

    val_loader = jdl.DataLoader(
        val_set,
        backend="jax",
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    return dataset, train_loader, val_loader
