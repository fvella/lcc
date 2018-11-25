# lcc
Serial and distributed code for local clustering coefficient

# Buiding LCC

To compile with foMPI load the dmapp module first:
```
module load dmapp
```

On Cray systems build with with:
```
module switch PrgEnv-cray PrgEnv-gnu

MPICC=cc MPICXX=CC ./configure <configure options>
```
