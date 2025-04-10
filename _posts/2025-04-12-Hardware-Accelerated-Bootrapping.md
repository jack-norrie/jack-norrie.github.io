
## Introduction

## Coverage Experiment

```python
def estimate_coverage(
    bootstrapper_constructor: Callable[..., Bootstrapper],
    sampling_distribution: SamplingDistribution,
    statistic_fn: StatisticFn,
    confidence_level: float = 0.95,
    n_sim: int = 10_000,
    n_samples: int = 100,
    n_boot: int = 2_000,
    batch_size: int = 1000,
    seed: int = 0,
):
    key = random.key(seed)

    key, subkey = random.split(key)
    true_statistic_value = statistic_fn(sampling_distribution(subkey, 1_000_000))

    bootstrapper = bootstrapper_constructor(statistic_fn)

    @jax.vmap
    @jax.jit
    def confidence_interval_simulation(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key, subkey = random.split(key)
        data = sampling_distribution(subkey, n_samples)

        key, subkey = random.split(key)
        bootstrapper.resample(data=data, n_resamples=n_boot, key=subkey)

        ci_low, ci_high = bootstrapper.ci(confidence_level=confidence_level)
        return (true_statistic_value >= ci_low) & (true_statistic_value <= ci_high), ci_high - ci_low

    covered_count = 0
    total_length = 0
    i = 0
    while i < n_sim:
        logging.debug(f"Batch: i / {n_sim}")

        current_batch_size = min(batch_size, n_sim - i)
        key, subkey = random.split(key)

        is_covered, length = confidence_interval_simulation(random.split(subkey, current_batch_size))
        covered_count += jnp.sum(is_covered)
        total_length += jnp.sum(length)

        i += batch_size
    coverage = covered_count / n_sim
    average_length = total_length / n_sim

    return coverage, average_length
```

## Simple Statistic

## Complex Statistic

```python
    def complex_distribution(key: jax.Array, n: int) -> jax.Array:
        # Log-normal distribution (heavily skewed)
        key, subkey = random.split(key)
        log_normal = jnp.exp(random.normal(subkey, shape=(n,)))

        # Add some contamination for extra complexity
        key, subkey = random.split(key)
        contamination = random.exponential(subkey, shape=(n,))

        return 0.9 * log_normal + 0.1 * contamination

    def complex_statistic(data: jax.Array) -> jax.Array:
        return jnp.mean(data) / (1 + jnp.median(data))
```

## Adversarial Percentile Setup

Now lets setup an example where the percentile bootstrapping method would be expected to struggle. Specifically we will bias the theta hat statistic by some multiple of the observed statistic variance, this is so that the bias is on an appropriate scale - else there will be numerical issues with inverse quantiles. Then we will bias the replicated statistic by twice this amount such that $\hat{\theta}^* - \hat{\theta}_n$ and $\hat{\theta}_n - \theta$ have the same offset. Under such a setup we would expect all the methods except the percentile method to do well, since all of these methods make pivotal assumptions based on the difference between true value and statistic.

```python
def resample(self, data: jax.Array, n_resamples: int = 2000, key: jax.Array = random.key(42)) -> None:
    key, subkey = random.split(key)

    self._theta_hat = self._statistic(data)

    @jax.vmap
    @jax.jit
    def _generate_bootstrap_replicate(rng_key: jax.Array) -> jax.Array:
        data_resampled = self._resample_data(data, rng_key)
        theta_boot = self._statistic(data_resampled)
        return theta_boot

    self._bootstrap_replicates = _generate_bootstrap_replicate(random.split(subkey, n_resamples))

    # Modification
    bias_factor = self.variance() * 2
    self._theta_hat = self._theta_hat + bias_factor
    self._bootstrap_replicates = self._bootstrap_replicates + 2 * bias_factor
```

```
