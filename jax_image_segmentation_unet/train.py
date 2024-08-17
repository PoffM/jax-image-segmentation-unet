import jax
import os

jax_cache_dir = os.path.join(os.path.dirname(__file__), ".jax_cache")

jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp
import flax.linen as nn
import optax  # type: ignore
from flax.training import train_state
from flax import struct
from clu import metrics  # type: ignore
from typing import cast
import orbax  # type: ignore
from flax.training import orbax_utils

import orbax.checkpoint  # type: ignore

from model import SegmentationUNet
from loader_loop import loader_loop
from init_loaders import init_loaders

checkpoint_path = os.path.join(os.path.dirname(__file__), ".trained_checkpoint")

print(f"JAX process: {jax.process_index() + 1} / {jax.process_count()}")
print(f"JAX local devices: {jax.local_devices()}")

key = jax.random.PRNGKey(0)

input_shape = (32, 28 * 4, 28 * 4, 1)
example_input = jnp.ones(input_shape)
seg_model = SegmentationUNet(in_channels=1, num_classes=10)


grid_wh = 4
batch_size = 32

dataset, train_loader, val_loader = init_loaders(grid_wh, batch_size)


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")  # type: ignore


class TrainState(train_state.TrainState):
    metrics: Metrics
    batch_stats: dict


def create_train_state(model: nn.Module, rng: jax.Array):
    """Creates an initial `TrainState`."""
    # initialize parameters by passing a template image
    variables = model.init(rng, example_input, train=False)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adamw(1e-3),
        metrics=Metrics.empty(),
        batch_stats=variables["batch_stats"],
    )


@jax.jit
def train_step(state: TrainState, batch: tuple[jax.Array, jax.Array]):
    """Train for a single step."""

    grayscale_input, true_colored = batch

    def loss_fn(params):
        preds, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            grayscale_input,
            train=True,
            mutable=["batch_stats"],
        )

        loss = segmentation_loss(preds, true_colored)

        return loss, (preds, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (preds, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    # Metrics
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)

    state = state.replace(
        batch_stats=updates["batch_stats"],
        metrics=metrics,
    )
    return state


@jax.jit
def compute_metrics(*, state: TrainState, batch: tuple[jax.Array, jax.Array]):
    grayscale_input, true_colored = batch

    preds, updates = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        grayscale_input,
        train=True,
        mutable=["batch_stats"],
    )

    loss = segmentation_loss(preds, true_colored)

    metric_updates = state.metrics.single_from_model_output(
        preds=preds, labels=true_colored, loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


def segmentation_loss(preds, true_colored):
    y = true_colored[:, :, :, :10]
    p = preds[:, :, :, :10]

    # binary cross entropy loss
    loss = (-(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))).mean()
    return loss


def train():
    init_rng = jax.random.key(0)

    state = create_train_state(seg_model, init_rng)
    del init_rng  # Must not be used anymore.

    epochs = 5

    num_steps_per_epoch = len(train_loader)

    metrics_history = {
        "train_loss": [],
        "validation_loss": [],
    }

    print("Start training loop")

    for epoch in range(epochs):
        for pbar, step, batch in loader_loop(train_loader, f"Epoch {epoch+1}/{epochs}"):
            # Run optimization steps over training batches and compute batch metrics
            state = train_step(
                state, batch
            )  # get updated train state (which contains the updated parameters)
            # state = compute_metrics(state=state, batch=batch)  # aggregate batch metrics

            state = cast(TrainState, state)

            pbar.set_postfix(
                {"train_loss": float(state.metrics.loss.compute_value().value)}
            )

            if (step + 1) % num_steps_per_epoch == 0:  # one training epoch has passed
                for metric, value in state.metrics.compute().items():  # compute metrics
                    metrics_history[f"train_{metric}"].append(value)  # record metrics
                state = state.replace(
                    metrics=state.metrics.empty()
                )  # reset train_metrics for next training epoch

                # Compute metrics on the val set after each training epoch
                val_state = state
                for val_pbar, val_step, val_batch in loader_loop(val_loader, f"Epoch {epoch+1}/{epochs} validation:"):
                    val_state = compute_metrics(state=val_state, batch=val_batch)

                for metric, value in val_state.metrics.compute().items():
                    metrics_history[f"validation_{metric}"].append(value)

                print(
                    f"train epoch: {(epoch+1)}, "
                    f"loss: {metrics_history['train_loss'][-1]}, "
                )
                print(
                    f"validation epoch: {(epoch+1)}, "
                    f"loss: {metrics_history['validation_loss'][-1]}, "
                )

    print(f"Training finished with loss of {metrics_history['train_loss']}")

    ckpt = {"model": state}

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(checkpoint_path, ckpt, save_args=save_args, force=True)


if __name__ == "__main__":
    train()
