import os
import pickle

import h5ify
import jax_tqdm
import numpy as np
import numpyro


# TODO: make serialization safe
def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def postprocess(file):
    z = h5ify.load(file)
    keys = list(z[list(z.keys())[0]].keys())
    z = {key: np.concatenate([z[i][key] for i in z]) for key in keys}
    h5ify.save(file, z, mode = 'w')
    return z


def run(
    label,
    kernel,
    num_warmup,
    num_samples,
    num_checkpoint,
    num_progress = 1,
    model_args = (),
    model_kwargs = {},
    rng_key = None,
    init_params = None,
):
    if os.path.exists(f'{label}.pkl'):
        state, i = load(f'{label}.pkl')
    else:
        assert rng_key is not None
        state = kernel.init(
            rng_key = rng_key,
            num_warmup = num_warmup,
            init_params = init_params,
            model_args = model_args,
            model_kwargs = model_kwargs,
        )
        i = 0

    body_fn = lambda state: kernel.sample(state, model_args, model_kwargs)
    postprocess_fn = kernel.postprocess_fn(model_args, model_kwargs)
    transform = lambda state: postprocess_fn(state.z)

    while i < num_warmup + num_samples:
        if i < num_warmup:
            length = min(num_warmup - i, num_checkpoint)
            pre = 'warmup'
        else:
            length = min(num_warmup + num_samples - i, num_checkpoint)
            pre = 'sample'
        desc = f'{i}-{i + length} / {num_warmup + num_samples} {pre}'

        z, state = numpyro.util.fori_collect(
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

        i += length

        print(f'checkpoint {i} / {num_warmup + num_samples}: {label}.pkl')
        save(f'{label}.pkl', (state, i))

        if i > num_warmup:
            print(f'samples {i - num_warmup} / {num_samples}: {label}.h5')
            h5ify.save(f'{label}.h5', {str(i - num_warmup): z})

    z = postprocess(f'{label}.h5')

    return state, i, z
