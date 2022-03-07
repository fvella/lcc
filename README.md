# LCC with RMA Caching

Serial, parallel and distributed code for the computation of the local clustering coefficient on large scale graphs.

Code base for the *Asynchronous Distributed-Memory Triangle Counting and LCC with RMA Caching [[1]](https://arxiv.org/abs/2202.13976!)* paper.
To reconstruct the results reported in the paper:
- compile with ```--with-simd --with-clampi``` options
- this version is compatible with the standard CLaMPI library ([git](https://github.com/spcl/CLaMPI)) and thus does not support user-specified eviction scores

## Buiding LCC

**General method:**

Generate configure script:
```
./autogen.sh
```

Configure and compile:
```
./configure <configure options>
make 
make install
```


**Building on Cray systems:**

Generate configure script:
```
./autogen.sh
```

To compile with foMPI load the dmapp module first:
```
module load dmapp
```

Configure:
```
module switch PrgEnv-cray PrgEnv-gnu

MPICC=cc MPICXX=CC ./configure <configure options>
make
make install
```

Type
```
./configure --help
```
for configuration options (LibLSB, CLaMPI, SIMD version, debug mode).

## Using LCC

For a description of the different options run ```./lcc```. The code assumes an adjacency list format if a graph file is used and skips comment lines starting with *%*.