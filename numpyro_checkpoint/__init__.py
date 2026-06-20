import os
import pickle

import jax
import jax.numpy as jnp
import jax_tqdm
import numpyro


# TODO: make serialization safe
# TODO: save new updates rather than whole chain every time
def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def init(
    file, kernel, rng_key, num_warmup, init_params, model_args, model_kwargs,
):
    if os.path.exists(file):
        state, z, i = load(file)
    else:
        state = kernel.init(
            rng_key = rng_key,
            num_warmup = num_warmup,
            init_params = init_params,
            model_args = model_args,
            model_kwargs = model_kwargs,
        )
        z = None
        i = 0
    return state, z, i


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
    state, z, i = state

    while i < num_warmup:
        length = min(num_warmup - i, num_checkpoint)

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
                desc = f'warmup {i}-{i + length} / {num_warmup}',
            )(fn)

        state, _ = jax.lax.scan(fn, state, jnp.arange(length))
        i += length

        save(file, (state, z, i))
        print(f'checkpoint {i} / {num_warmup}: {file}')

    return state, z, i


## TODO: use fori_collect with progbar instead?
def sample(
    file,
    kernel,
    state,
    num_warmup,
    num_samples,
    num_checkpoint,
    num_progress,
    model_args,
    model_kwargs,
):
    state, z, i = state

    while i < num_warmup + num_samples:
        length = min(num_warmup + num_samples - i, num_checkpoint)
        start = i - num_warmup
        stop = i - num_warmup + length

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
                desc = f'sample {start}-{stop} / {num_samples}',
            )(fn)

        state, new_z = jax.lax.scan(fn, state, jnp.arange(length))

        postprocess_fn = kernel.postprocess_fn(model_args, model_kwargs)
        fn = lambda _, iz: (None, postprocess_fn(iz[1]))

        if num_progress is not None:
            fn = jax_tqdm.scan_tqdm(
                length,
                print_rate = num_progress,
                tqdm_type = 'std',
                desc = f'postprocess {start}-{stop}',
            )(fn)

        _, new_z = jax.lax.scan(fn, None, (jnp.arange(length), new_z))

        if z is None:
            z = new_z
        else:
            z = {key: jnp.concatenate([z[key], new_z[key]]) for key in z}

        i += length

        save(file, (state, z, i))
        print(f'checkpoint {i} / {num_warmup + num_samples}: {file}')

    return state, z, i


def run(
    file,
    kernel,
    num_warmup,
    num_samples,
    num_checkpoint,
    num_progress = None,
    rng_key = None,
    model_args = (),
    model_kwargs = {},
):
    init_params = None
    state = init(
        file,
        kernel,
        rng_key,
        num_warmup,
        init_params,
        model_args,
        model_kwargs,
    )
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
    state = sample(
        file,
        kernel,
        state,
        num_warmup,
        num_samples,
        num_checkpoint,
        num_progress,
        model_args,
        model_kwargs,
    )
    return state


def _run(
    file,
    kernel,
    model_args,
    model_kwargs,
    rng_key,
    init_params,
    num_warmup,
    num_samples,
    num_checkpoint,
    num_progress,
):
    if os.path.exists(file):
        state, z, i = load(file)
    else:
        state = kernel.init(
            rng_key = rng_key,
            num_warmup = num_warmup,
            init_params = init_params,
            model_args = model_args,
            model_kwargs = model_kwargs,
        )
        z = None
        i = 0

    body_fn = lambda state: kernel.sample(state, model_args, model_kwargs)
    postprocess_fn = kernel.postprocess_fn(model_args, model_kwargs)
    transform = lambda state: postprocess_fn(state.z)

    while i < num_warmup + num_samples:
        desc = f'{i}-{i + length} / {num_warmup + num_samples}'
        if i < num_warmup:
            length = min(num_warmup - i, num_checkpoint)
            desc += ' warmup'
        else:
            length = min(num_warmup + num_samples - i, num_checkpoint)
            desc += ' sample'

        new_z, state = numpyro.util.fori_collect(
            lower = 0,
            upper = length,
            body_fun = body_fn,
            init_val = state,
            transform = transform,
            progbar = False if num_progress is None else True,
            progress_rate = num_progress,
            return_last_val = True,
            # collection_size = None,
            # thinning = 1,
            # **progbar_opts,
            progbar_desc = lambda i: desc,
            diagnostics_fn = kernel.get_diagnostics_str,
            # num_chains = 1,
        )

        if i < num_warmup:
            new_z = None

        if z is None:
            z = new_z
        else:
            z = {key: jnp.concatenate([z[key], new_z[key]]) for key in z}

        i += length

        print(f'checkpoint {i} / {num_warmup + num_samples}: {file}')
        save(file, (state, z, i))

    return state, z, i
