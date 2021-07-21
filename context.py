import copy
import typing

from jax import numpy as jnp, random

import optax
from optax._src import transform

def gpt3_schedule(warmup_steps,
                  total_steps,
                  peak_lr,
                  end_lr):
    def sch(step):
        warmup_pct = jnp.clip(step, 0, warmup_steps) / warmup_steps
        anneal_pct = jnp.clip(step - warmup_steps, 0, total_steps) / total_steps

        return warmup_pct * peak_lr - (peak_lr - end_lr) * (1 - jnp.cos(jnp.pi * anneal_pct)) / 2

    return sch


class DataContext:
    def __init__(self):
        self.path = "gs://obst-euw4a-aa/the-char-pile/*"
        #self.shuffle_buffer = 2 ** 16
        self.shuffle_buffer = 2 ** 8
        self.parallel_interleave = True
        self.interleaved_datasets = 8
        self.seed = 0
        self.vocab_size = 256  # should be divisible by 128
        self.path = "gs://obst-euw4a-aa/the-bpe-pile/*"
        self.vocab_size = 50304


class Dims:
    def __init__(self, data: DataContext, group_linear_factor):
        self.batch = "batch"
        self.features_per_head = "features_per_head"
        self.heads = "heads"
        self.sequence = "sequence"
        self.intermediate_feed_forward = "intermediate_feed_forward"
        self.one = "one"
        self.vocab = "vocab"
        self.dim_sizes: typing.Dict[str, int] = {self.batch: 1,
                                                 self.features_per_head: 64,
                                                 self.heads: 8,
                                                 self.sequence: 1024,
                                                 self.vocab: data.vocab_size,
                                                 self.one: 1}
        self.dim_sizes[self.intermediate_feed_forward] = self.dim_sizes[self.features_per_head] * group_linear_factor

    def get_dim_size(self, dim):
        if isinstance(dim, str):
            dim = self.dim_sizes[dim]
        return dim


class Context:
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.seed = 0
        self.prng_key = random.PRNGKey(self.seed)
        self.learning_rate = 1.2e-4
        self.end_learning_rate = 1.2e-5
        self.warmup_steps = 3000
        self.anneal_steps = 300000
        # self.learning_rate = 1e-3
        self.weight_decay = 0.1
        # self.opt = optax.adam(self.learning_rate)
        self.opt = optax.chain(
            # optax.scale(1 / gradient_accumulation_steps),
            transform.clipping.clip_by_global_norm(1),
            transform.scale_by_adam(),
            # transform.scale_by_sm3(),
            # transform.scale(120/10),
            transform.additive_weight_decay(self.weight_decay),
            transform.scale(-1),
            # transform.scale(self.learning_rate),
            optax.scale_by_schedule(gpt3_schedule(self.warmup_steps, self.anneal_steps, self.learning_rate, self.end_learning_rate))
        )

        self.parameters: typing.Dict[str, jnp.ndarray] = {}
        self.parameter_dims: typing.Dict[str, typing.List[str]] = {}
        self.device_steps = 2 ** 4
        self.steps = 2 ** 32
        self.head_count = 1
        self.norm_eps = 1e-5
        self.group_linear_factor = 2
        self.depth = 16
        self.dtype = jnp.float32
        # self.init_scale = 1.0
        self.init_scale = 0.02
        self.global_prefix = ''
        self.model_parallel = 8
        self.data_parallel = 1
        self.z_loss = 1e-5
        # self.embedding_std = 0.004
        self.embedding_std = 0.02
        self.norm_std = 0
        self.dense_std = 0.02
        self.name_cache: typing.Dict[str, int] = {}
        self.masked_attention = True
        self.print_interval = 1
        self.data = DataContext()
        self.dims = Dims(self.data, self.group_linear_factor)

        if config is not None:
            self.__dict__.update(config)

    def add_to_prefix(self, appended=""):
        new = copy.copy(self)
        new.global_prefix = self.global_prefix + '/' + self.incremental_name(appended)
        new.prng_key = random.fold_in(new.prng_key, hash(new.global_prefix))
        new.name_cache = {}
        return new

    def incremental_name(self, name):
        if name not in self.name_cache:
            self.name_cache[name] = -1
        self.name_cache[name] += 1
        return f'{name}:{self.name_cache[name]:d}'

    def init(self):
        self.opt_state = self.opt.init(self.parameters)
        return self.parameters, self.opt_state

    def next_prng_key(self):
        self.prng_key, next_key = random.split(self.prng_key)
        return next_key


class WhileContext:
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.ctx = Context()
        self.current_step = jnp.zeros([], dtype=jnp.uint32)
        self.data: typing.Optional[jnp.ndarray] = None
        self.loss = jnp.zeros([])

        if config is not None:
            self.ctx.parameters = config['parameters']
            self.ctx.opt_state = config['opt_state']
            self.loss = config['loss']
            self.current_step = config['current_step']
            self.data = config['data']

    def serialize(self):
        return {'parameters': self.ctx.parameters,
                'opt_state': self.ctx.opt_state,
                'current_step': self.current_step,
                'loss': self.loss,
                'data': self.data}
