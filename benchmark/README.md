# Benchmark
-----------

TTC provides a [benchmark for tensor transpositions](https://github.com/HPAC/TTC/blob/master/benchmark/benchmark.py).

    python benchmark.py <num_threads>

This will generate the input strings for TTC for each of the test-cases within the benchmark. 
The benchmark uses a default tensor size of 200 MiB (see _sizeMB variable)
