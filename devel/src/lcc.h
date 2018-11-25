#ifndef _LCC_H_
#define _LCC_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define OVERLAP 1
#define ONEPREFIX 1
#define THRUST 1
//#define _LARGE_LVERTS_NUM
#ifdef    _LARGE_LVERTS_NUM
//#define LOCINT       uint64_t
#define LOCINT       uint64_t
#define LOCPRI       PRIu64
#define LOCINT_MPI MPI_UNSIGNED_LONG_LONG
#else
#define LOCINT       uint32_t
#define LOCPRI       PRIu32
#define LOCINT_MPI MPI_UNSIGNED
#endif
#define DEFAULT_SEED 4343543
#define MAX(a,b)    (((a)>(b))?(a):(b))
#define MIN(a,b)    (((a)<(b))?(a):(b))

#define MAX_PROC_I    1024
#define MAX_PROC_J    1024
#define MAX_LINE    1024
#define MAX_LVL     1000

//#define    _64(i)        ((uint64_t)(i))
#if 1
#define GI2PI(i)    (((i)%col_bl)/row_bl)        // global row -> processor row
#define GJ2PJ(j)    ((j)/col_bl)                // global col -> processor col

// Given the edge i,j returns the processor id owning the edge
#define EDGE2PROC(i,j)    ((int)(GI2PI(i)*C + GJ2PJ(j)))    // global (i,j) -> processor

// Given the vertex i, returns the processor owning the edge (i,i)
#define VERT2PROC(i)    (EDGE2PROC(i,i))        // global vertex (i,)  -> processor

#define GI2LOCI(i)    (((i)/col_bl)*row_bl + (i)%row_bl) // global row -> local row
#define GJ2LOCJ(j)    ((j)%col_bl)               // global col -> local col

#define LOCI2GI(i)    (((i)/row_bl)*col_bl + myrow*row_bl + (i)%row_bl) // local row -> global row
#define LOCJ2GJ(j)    (mycol*col_bl + (j))                  // local col -> global col

#define    LIJ2LN(i,j)    ((i)*col_bl + (j))        // local  edge (i,j) to  local linear index
#define    LIJ2GN(i,j)    (LOCI2GI(i)*N + LOCJ2GJ(j)) // local  edge (i,j) to global linear index
#define GIJ2GN(i,j)    ((i)*N + (j))            // global edge (i,j) to global linear index

#define LIJ2PROC(i,j) (EDGE2PROC(LOCI2GI(i),LOCJ2GJ(j)))

#define    GN2GI(n)    ((n)/N) // global linear index to global (i,)
#define    GN2GJ(n)    ((n)%N) // global linear index to global (,j)

#define    GN2LI(n)    (GI2LOCI(GN2GI(n))) // global linear index to local (i,)
#define    GN2LJ(n)    (GJ2LOCJ(GN2GJ(n))) // global linear index to local (,j)

#define MYLOCI2LOCJ(i)    (((i)%row_bl) + myrow*row_bl)    // local OWNED row -> local col
#define MYLOCJ2LOCI(j)    (((j)%row_bl) + mycol*row_bl)    // local OWNED col -> local row
#define REMJ2GJ(j,pj)    ((pj)*col_bl + (j))        // col of proc (*,pj) -> global col

#define ISMYCOLL(j)        ((j/row_bl) == (LOCINT)(myrow))  // local col -> proc row

//#define CUDA_MYLOCI2LOCJ(i)  (((i)%drow_bl) + dmyrow*drow_bl)
#define CUDA_MYLOCI2LOCJ(i)  ((i)+(dmyrow-((i)/drow_bl))*drow_bl)
#define CUDA_ISMYCOLL(j)     (((j)/drow_bl) == (LOCINT)(dmyrow))  // local col -> proc row
#define CUDA_ISMYROW(i)         (((i)/dcol_bl) == (LOCINT)(dmycol))  // local row -> proc row
#define CUDA_MYLOCJ2LOCI(j)  ((j)+(dmycol-((j)/drow_bl))*drow_bl)    // local OWNED col -> local row

#else

#define GI2PI(i)    ((i)%R)                // global row -> processor row
#define GJ2PJ(j)    ((j)/col_bl)  //((j)%C)        // global col -> processor col
#define EDGE2PROC(i,j)    ((int)(GI2PI(i)*C + GJ2PJ(j)))    // global (i,j) -> processor
#define VERT2PROC(i)    (EDGE2PROC(i,i))        // global vertex (i,)  -> processor

#define GI2LOCI(i)    ((i)/R)                // global row -> local row
#define GJ2LOCJ(j)    ((j)%col_bl)    //((j)/C)    // global col -> local col

#define LOCI2GI(i)    ((i)*R + myrow)    // local row -> global row
#define LOCJ2GJ(j)    (mycol*col_bl + (j)) //((j)*C + mycol)    // local col -> global col
/* UNUSED, to be modified
#define    LIJ2LN(i,j)    ((i)*col_bl + (j))        // local  edge (i,j) to  local linear index
#define    LIJ2GN(i,j)    (LOCI2GI(i)*N + LOCJ2GJ(j)) // local  edge (i,j) to global linear index
#define GIJ2GN(i,j)    ((i)*N + (j))            // global edge (i,j) to global linear index

#define    GN2GI(n)    ((n)/N) // global linear index to global (i,)
#define    GN2GJ(n)    ((n)%N) // global linear index to global (,j)

#define    GN2LI(n)    (GI2LOCI(GN2GI(n))) // global linear index to local (i,)
#define    GN2LJ(n)    (GJ2LOCJ(GN2GJ(n))) // global linear index to local (,j)
*/
#define MYLOCI2LOCJ(i)    (((i)%row_bl)*R + myrow)         // local OWNED row -> local col
#define REMJ2GJ(j,pj)    ((pj)*col_bl + (j))    //((j)*C + (pj))    // col of proc (*,pj) -> global col

#define CUDA_MYLOCI2LOCJ(i)  (((i)%drow_bl)*dR + dmyrow)

#endif ///////////////

#define BITS(ptr)    (sizeof(*ptr)<<3)
#define MSKGET(mask,n)  (mask[(n)/BITS(mask)] &  (1ULL<<((n)%BITS(mask)))) // get mask bit for vert at CSR index n
#define MSKSET(mask,n)  {mask[(n)/BITS(mask)] |= (1ULL<<((n)%BITS(mask)));} // set mask bit for vert at CSR index n

#define TIMER_DEF(n)     struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)     gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)     gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)*1.e6+(temp_2_##n.tv_usec-temp_1_##n.tv_usec))

typedef struct
{
    uint64_t id;         // Vertex identifier
    LOCINT degree;      // Vertex degree
    int lvl;            // Levels explored
    uint64_t msgSize;   // MPI message size
    uint64_t visited;   // Vertices visited
    uint64_t over;      // overlap time
    uint64_t compu;      // Computation Total time
    uint64_t commu;      // Communication Total time
    uint64_t tot;       // Total time
    uint64_t upw[2];    // Computation and Communication time
    uint64_t dep[2];    // Computation and Communication time
    uint64_t upd[2];    // Computation and Communication time
    LOCINT nfrt[MAX_LVL];    // For each level how many elements in the frontier

} STATDATA;



#ifdef __cplusplus
#define LINKAGE "C"
#else
#define LINKAGE
#endif
extern LINKAGE void init_bc_1degree_device(LOCINT *reach);

extern LINKAGE void *Malloc(size_t sz);
extern LINKAGE uint64_t compact(uint64_t *v, uint64_t ld, int *vnum, int n); 
extern LINKAGE void prstat(uint64_t val, const char *msg, int det);
extern LINKAGE void *CudaMallocHostSet(size_t size, int val); 
extern LINKAGE void sort_by_degree(LOCINT *deg,LOCINT *bc_order);
extern LINKAGE void dump_array2(int *arr, int n, const char *name);
extern LINKAGE void dump_uarray2(LOCINT *arr, int n, const char *name);
extern LINKAGE void dump_farray2(float *arr, int n, const char *name);

//extern LOCINT N;    /* number of vertices: N */
extern uint64_t    N;    /* number of vertices: N */
extern LOCINT    row_bl; /* adjacency matrix rows per block: N/(RC) */
extern LOCINT    col_bl; /* adjacency matrix columns per block: N/C */
extern LOCINT    row_pp; /* adjacency matrix rows per proc: N/(RC) * C = N/R */

extern int C;
extern int R;
extern int myid;
extern int ntask;
extern int mono;
extern int heuristic;
extern int myrow;
extern int mycol;
extern int pmesh[MAX_PROC_I][MAX_PROC_J];
extern MPI_Comm Row_comm, Col_comm;
extern FILE *outdebug;
extern LOCINT *tlvl;
extern LOCINT *tlvl_v1;
extern uint64_t overlap_time; 

#endif
