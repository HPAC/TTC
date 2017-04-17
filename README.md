# Tensor Transpose Compiler #

The Tensor Transpose Compiler (TTC) generates high-performance parallel and vectorized C++ code for multidimensional tensor transpositions.

TTC supports arbitrarily dimensional, out-of-place tensor transpositions of the general form:

![ttc](https://github.com/HPAC/TTC/blob/master/misc/equation.png)

where A and B respectively denote the input and output tensor;
<img src=https://github.com/HPAC/TTC/blob/master/misc/pi.png height=16px/> represents the user-specified
transposition, and 
<img src=https://github.com/HPAC/TTC/blob/master/misc/alpha.png height=14px/> and
<img src=https://github.com/HPAC/TTC/blob/master/misc/beta.png height=16px/> being scalars
(i.e., setting <img src=https://github.com/HPAC/TTC/blob/master/misc/beta.png height=16px/> != 0 enables the user to update the output tensor B).

Please also have a look at [TTC-C](https://github.com/HPAC/TTC-C) which provides
a C wrapper API for TTC.

Current version: **v0.1.1**

# Key Features
--------------

* Generates parallelized and vectorized code
* Support for multiple instruction sets: **AVX, AVX2, AVX512, KNC, CUDA**
* Support for all datatypes (i.e., single, double, single-complex and double-complex)
* Support for **mixed precision** (i.e., different data types for A and B)
    * For instance, this feature can be used to generate a mixed-precision BLAS
* Support for multiple leading dimensions in both A and B
    * This enables the user to extract (and transpose) a smaller subtensor out of a larger tensor
* TTC allows the user to guide the code generation process:
    * E.g., specifying the --maxImplementations=N argument will limited the number of generated implementations to N


# Install
---------

1. Clone the repository into a desired directory and change to that location:

    git clone https://github.com/HPAC/TTC.git
    cd TTC

2. Install TTC:

    python setup.py install --user

3. Make sure that you export the TTC_ROOT environment variable (add to your .bashrc):

    export TTC_ROOT=`pwd`

4. You might have to add the installed location to your PATH environment variable:
   
    export PATH=$PATH:~/.local/bin


# Getting Started
-----------------

Please run **ttc --help** to get an overview of TTC's parameters.

Here is one exemplary input to TTC: 

    ttc --perm=1,0,2 --size=1000,768,16 --dataType=s --alpha=1.0 --beta=1.0 --numThreads=20

# Requirements
--------------

You must have a working C compiler. I have tested TTC with:

* Python (tested with v2.7.5 and v2.7.9)
* GCC (>= 4.8)
* Intel's ICC (>= 15.0)

# Benchmark
-----------

TTC provides a [benchmark for tensor transpositions](https://github.com/HPAC/TTC/blob/master/benchmark/benchmark.py).

    python benchmark.py <num_threads>

This will generate the input strings for TTC for each of the test-cases within the benchmark. 
The benchmark uses a default tensor size of 200 MiB (see _sizeMB variable)


# Citation
-----------

In case you want refer to TTC as part of a research paper, please cite the following
article [(pdf)](http://arxiv.org/abs/1603.02297):
```
@article{ttc2016a,
   author      = {Paul Springer and Jeff R. Hammond and Paolo Bientinesi},
   title       = {{TTC}: A high-performance Compiler for Tensor Transpositions},
   archivePrefix = "arXiv",
   eprint = {1603.02297},
   primaryClass = "quant-ph",
   journal     = {CoRR},
   year        = {2016},
   issue_date  = {March 2016},
   url         = {http://arxiv.org/abs/1603.02297}
}
``` 

# Changelog
-----------
V0.1.1:
   * Improved performance for streaming stores (only applicable if beta=0)



# Feedback & Contributions
-----------

We are happy for any feedback or feature requests. Please contact springer@aices.rwth-aachen.de.

We also welcome any contributions to the code base.
