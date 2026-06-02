import os
import pickle

import h5ify
import jax
import jax.numpy as jnp
import jax_tqdm
import numpyro


def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def init(file, kernel, key, num_warmup, model_args, model_kwargs):
    if os.path.exists(file):
        state, z = load(file)
    else:
        state = kernel.init(
            key,
            num_warmup = num_warmup,
            model_args = model_args,
            model_kwargs = model_kwargs,
        )
        z = None
    return state, z


def warmup(
    file,
    kernel,
    state,
    num_warmup,
    num_checkpoint,
    num_progress,
    model_args,
    model_kwargs,
):
    left = num_warmup - state.i

    while left > 0:
        length = int(min(left, num_checkpoint))
        
        fn = lambda state, i: (
            kernel.sample(
                state, model_args = model_args, model_kwargs = model_kwargs,
            ),
            None,
        )

        if num_progress is not None:
            fn = jax_tqdm.scan_tqdm(
                length,
                print_rate = num_progress,
                tqdm_type = 'std',
                desc = f'warmup {left} / {num_warmup}',
            )(fn)

        state, _ = jax.lax.scan(fn, state, jnp.arange(length))
        save(file, (state, None))
        left -= length

    return state


def sample(
    file,
    kernel,
    state,
    z,
    num_warmup,
    num_samples,
    num_checkpoint,
    num_progress,
    model_args,
    model_kwargs,
):
    left = num_warmup + num_samples - state.i

    while left > 0:
        length = int(min(left, num_checkpoint))

        fn = lambda state, i: (
            kernel.sample(
                state, model_args = model_args, model_kwargs = model_kwargs,
            ),
            state.z,
        )
        
        if num_progress is not None:
            fn = jax_tqdm.scan_tqdm(
                length,
                print_rate = num_progress,
                tqdm_type = 'std',
                desc = f'sample {left} / {num_samples}',
            )(fn)

        state, new_z = jax.lax.scan(fn, state, jnp.arange(length))

        new_z = numpyro.infer.util.constrain_fn(
            kernel.model, model_args, model_kwargs, new_z, return_deterministic = True,
        )

        if z is None:
            z = new_z
        else:
            z = {key: jnp.concatenate([z[key], new_z[key]]) for key in z}

        save(file, (state, z))

        left -= length

    return state, z


def run(
    file,
    kernel,
    num_warmup,
    num_samples,
    num_checkpoint,
    key = None,
    model_args = (),
    model_kwargs = {},
    num_progress = None,
    postprocess = False,
):
    state, z = init(file, kernel, key, num_warmup, model_args, model_kwargs)

    state = warmup(
        file,
        kernel,
        state,
        num_warmup,
        num_checkpoint,
        num_progress,
        model_args,
        model_kwargs,
    )

    state, z = sample(
        file,
        kernel,
        state,
        z,
        num_warmup,
        num_samples,
        num_checkpoint,
        num_progress,
        model_args,
        model_kwargs,
    )
        
    return state, z
