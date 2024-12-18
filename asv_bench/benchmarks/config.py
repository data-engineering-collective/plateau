import timeit


class AsvBenchmarkConfig:
    """A mixin class to define default configurations for our benchmarks."""

    # Use a timer which measured Wall time (asv default measures CPU time)
    timer = timeit.default_timer
