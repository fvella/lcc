# Lcc
Serial, parallel and distributed code for the computation of the local clustering coefficient on large scale graphs.

# Buiding LCC

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
for more configuration options (e.g., LibLSB, CLaMPI).

