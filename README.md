# lcc
Serial and distributed code for local clustering coefficient

# Buiding LCC

Generate configure script:
```
./autogen.sh
./configure <configure options>
make 
make install
```


**On Cray systems:**

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

