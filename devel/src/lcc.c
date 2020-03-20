#include "lcc.h"
#include <dlfcn.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#define GRAPH_GENERATOR_MPI
#include "../../generator/make_graph.h"

#include "config.h"

#define RUNS 1
#define WARMUP 0
#include "mpi_wrapper.h"

#ifdef HAVE_LIBLSB
#include "liblsb.h"
#endif

#ifdef HAVE_CLAMPI
#include "clampi.h"
#endif

#ifdef HAVE_SIMD
#include <omp.h>
#else
#define SERIAL
#endif

#define VTAG(t) (0 * ntask + (t))
#define HTAG(t) (100 * ntask + (t))
#define PTAG(t) (200 * ntask + (t))

#define _TIMINGS 1
//#define PRINTSTATS 1
#define BIN_NUM 16

#define LIMIT 1
uint64_t N = 0; /* number of vertices: N */
LOCINT row_bl;  /* adjacency matrix rows per block: N/(RC) */
LOCINT col_bl;  /* adjacency matrix columns per block: N/C */
LOCINT row_pp;  /* adjacency matrix rows per proc: N/(RC) * C = N/R */
uint64_t degree_reduction_time = 0;
uint64_t sort_time = 0;
int C = 0;
int R = 0;
int gmyid;
int myid;
int gntask;
int ntask;
static int nthreads=1;
int mono = 1;
int undirected = 1;
int analyze_degree = 0;

int heuristic = 0;

int myrow;
int mycol;
int pmesh[MAX_PROC_I][MAX_PROC_J];
MPI_Comm MPI_COMM_CLUSTER;

LOCINT flag = 0;

LOCINT *reach = NULL;
FILE *outdebug = NULL;
LOCINT loc_count = 0;
STATDATA *mystats = NULL;
unsigned int outId;
char strmesh[10];
char cmdLine[256];
static size_t tot_dev_mem = 0;

MPI_Comm Row_comm, Col_comm;

static void freeMem(void *p) {
  if (p)
    free(p);
}

static void prexit(const char *fmt, ...) {

  int myid;
  va_list ap;

  va_start(ap, fmt);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (0 == myid)
    vfprintf(stderr, fmt, ap);
  MPI_Finalize();
  exit(EXIT_FAILURE);
}

void *Malloc(size_t sz) {

  void *ptr;

  ptr = (void *)malloc(sz);
  if (!ptr) {
    fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
    exit(EXIT_FAILURE);
  }
  memset(ptr, 0, sz);
  return ptr;
}

// root node, degree, nvisited, num_lvl

/*
 * Print statistics
 */
void prstat(uint64_t val, const char *msg, int det) {

  int myid, ntask, i, j, w1, w2, min, max;
  uint64_t t, *v = NULL;
  double m, s;

  MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);
  MPI_Comm_size(MPI_COMM_CLUSTER, &ntask);

  if (myid == 0)
    v = (uint64_t *)Malloc(ntask * sizeof(*v));

  MPI_Gather(&val, 1, MPI_UNSIGNED_LONG_LONG, v, 1, MPI_UNSIGNED_LONG_LONG, 0,
             MPI_COMM_CLUSTER);

  if (myid == 0) {
    t = 0;
    m = s = 0.0;
    min = max = 0;
    for (i = 0; i < ntask; i++) {
      if (v[i] < v[min])
        min = i;
      if (v[i] > v[max])
        max = i;
      t += v[i];
      m += (double)v[i];
      s += (double)v[i] * (double)v[i];
    }
    m /= ntask;
    s = sqrt((1.0 / (ntask - 1)) * (s - ntask * m * m));

    fprintf(stdout, "%s", msg);
    if (det) {
      for (w1 = 0, val = ntask; val; val /= 10, w1++)
        ;
      for (w2 = 0, val = v[max]; val; val /= 10, w2++)
        ;

      fprintf(stdout, "\n");
      for (i = 0; i < R; i++) {
        fprintf(stdout, " ");
        for (j = 0; j < C; j++) {
          fprintf(stdout, "%*d: %*" PRIu64 "  ", w1, i * C + j, w2,
                  v[i * C + j]);
        }
        fprintf(stdout, "\n");
      }
    }
    fprintf(stdout,
            " [total=%" PRIu64 ", mean=%.2lf, stdev=%.2lf, min(%d)=%" PRIu64
            ", max(%d)=%" PRIu64 "]\n",
            t, m, s, min, v[min], max, v[max]);

    free(v);
  }
  return;
}

static void *Realloc(void *ptr, size_t sz) {

  void *lp;

  lp = (void *)realloc(ptr, sz);
  if (!lp && sz) {
    fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
    exit(EXIT_FAILURE);
  }
  return lp;
}

static FILE *Fopen(const char *path, const char *mode) {

  FILE *fp = NULL;
  fp = fopen(path, mode);
  if (!fp) {
    fprintf(stderr, "Cannot open file %s...\n", path);
    exit(EXIT_FAILURE);
  }
  return fp;
}
static off_t get_fsize(const char *fpath) {

  struct stat st;
  int rv;

  rv = stat(fpath, &st);
  if (rv) {
    fprintf(stderr, "Cannot stat file %s...\n", fpath);
    exit(EXIT_FAILURE);
  }
  return st.st_size;
}

static uint64_t getFsize(FILE *fp) {

  int rv;
  uint64_t size = 0;

  rv = fseek(fp, 0, SEEK_END);
  if (rv != 0) {
    fprintf(stderr, "SEEK END FAILED\n");
    if (ferror(fp))
      fprintf(stderr, "FERROR SET\n");
    exit(EXIT_FAILURE);
  }

  size = ftell(fp);
  rv = fseek(fp, 0, SEEK_SET);

  if (rv != 0) {
    fprintf(stderr, "SEEK SET FAILED\n");
    exit(EXIT_FAILURE);
  }

  return size;
}
/*
 * Duplicates vertices to make graph undirected
 *
 */
static uint64_t *mirror(uint64_t *ed, uint64_t *ned) {

  uint64_t i, n;

  if (undirected == 1) {
    ed = (uint64_t *)Realloc(ed, (ned[0] * 4) * sizeof(*ed));

    n = 0;
    for (i = 0; i < ned[0]; i++) {
      if (ed[2 * i] != ed[2 * i + 1]) {
        ed[2 * ned[0] + 2 * n] = ed[2 * i + 1];
        ed[2 * ned[0] + 2 * n + 1] = ed[2 * i];
        n++;
      }
    }
    ned[0] += n;
  }
  return ed;
}

/*
 * Read graph data from file
 */
static uint64_t read_graph(int myid, int ntask, const char *fpath,
                           uint64_t **edge) {
#define ALLOC_BLOCK (2 * 1024)

  uint64_t *ed = NULL;
  uint64_t i, j;
  uint64_t n, nmax;
  uint64_t size;
  int64_t off1, off2;

  int64_t rem;
  FILE *fp;
  char str[MAX_LINE];

  fp = Fopen(fpath, "r");

  size = getFsize(fp);
  rem = size % ntask;
  off1 = (size / ntask) * myid + ((myid > rem) ? rem : myid);
  off2 = (size / ntask) * (myid + 1) + (((myid + 1) > rem) ? rem : (myid + 1));

  if (myid < (ntask - 1)) {
    fseek(fp, off2, SEEK_SET);
    fgets(str, MAX_LINE, fp);
    off2 = ftell(fp);
  }
  fseek(fp, off1, SEEK_SET);
  if (myid > 0) {
    fgets(str, MAX_LINE, fp);
    off1 = ftell(fp);
  }

  n = 0;
  nmax = ALLOC_BLOCK; // must be even
  ed = (uint64_t *)Malloc(nmax * sizeof(*ed));
  uint64_t lcounter = 0;
  uint64_t nedges = -1;
  int comment_counter = 0;

  /* read edges from file */
  while (ftell(fp) < off2) {

    // Read the whole line
    fgets(str, MAX_LINE, fp);

    // Strip # from the beginning of the line
    if (strstr(str, "#") != NULL) {
      // fprintf(stdout, "\nreading line number %"PRIu64": %s\n", lcounter,
      // str);
      if (strstr(str, "Nodes:")) {
        sscanf(str, "# Nodes: %" PRIu64 " Edges: %" PRIu64 "\n", &i, &nedges);
        fprintf(stdout, "N=%" PRIu64 " E=%" PRIu64 "\n", i, nedges);
      }
      comment_counter++;
    } else if (str[0] != '\0') {
      lcounter++;
      // Read edges
      sscanf(str, "%" PRIu64 " %" PRIu64 "\n", &i, &j);

      if (i >= N || j >= N) {
        fprintf(stderr, "[%d] In file %s line %" PRIu64
                        " found invalid edge in %s for N=%" PRIu64 ": (%" PRIu64
                        ", %" PRIu64 ")\n",
                myid, fpath, lcounter, str, N, i, j);
        exit(EXIT_FAILURE);
      }

      if (n >= nmax) {
        nmax += ALLOC_BLOCK;
        ed = (uint64_t *)Realloc(ed, nmax * sizeof(*ed));
      }
      ed[n] = i;
      ed[n + 1] = j;
      n += 2;
    }
  }
  // Check the number of edges against the number of lines, if there were
  // comments
  // nedges += comment_counter;
  /*
  if ((comment_counter > 0) && (lcounter != nedges)) {
      fprintf(stderr, "Error reading the input file %s: the number of lines
  differ from the number of edges in the header\n", fpath);
      fprintf(stderr, "lcounter = %"PRIu64" nedges = %"PRIu64"\n", lcounter,
  nedges);
      exit(EXIT_FAILURE);
  }
   */
  fclose(fp);

  n /= 2; // number of ints -> number of edges
  *edge = mirror(ed, &n);
  return n;
#undef ALLOC_BLOCK
}

/*
 *  select one root vertex randomly
 */

/*
 * Generate RMAT graph calling make_graph function
 */
static uint64_t gen_graph(int scale, int edgef, uint64_t **ed) {

  uint64_t ned;
  double initiator[4] = {.57, .19, .19, .05};

  make_graph(scale, (((int64_t)1) << scale) * edgef, 23, 24, initiator,
             (int64_t *)&ned, (int64_t **)ed, MPI_COMM_CLUSTER);
  *ed = mirror(*ed, &ned);

  return ned;
}

static void dump_edges(uint64_t *ed, uint64_t nedge, const char *desc) {

  uint64_t i;
  fprintf(outdebug, "%s - %ld\n", desc, nedge);

  for (i = 0; i < nedge; i++)
    fprintf(outdebug, "%" PRIu64 "\t%" PRIu64 "\n", ed[2 * i], ed[2 * i + 1]);

  fprintf(outdebug, "\n");
  return;
}

static int cmpedge_1d(const void *p1, const void *p2) {
  uint64_t *l1 = (uint64_t *)p1;
  uint64_t *l2 = (uint64_t *)p2;
  if (l1[0] < l2[0])
    return -1;
  if (l1[0] > l2[0])
    return 1;
  if (l1[1] < l2[1])
    return -1;
  if (l1[1] > l2[1])
    return 1;
  return 0;
}
static void dump_rmat(uint64_t *myedges, uint64_t myned, int scale, int edgef) {

  FILE *fout = NULL;
  char fname[256];

  uint64_t i;
  uint64_t max = 0;

  qsort(myedges, myned, sizeof(uint64_t[2]), cmpedge_1d);
  snprintf(fname, 256, "rmat_S%d_EF%d.txt", scale, edgef);
  fprintf(stdout, "DUMP THE GENERATED RMAT FILE IN %s\n", fname);
  fout = fopen(fname, "w+");
  if (fout == NULL) {
    fprintf(stderr, "in function %s: error opening %s\n", __func__, fname);
    exit(EXIT_FAILURE);
  }
  max = N;
  fflush(stdout);
  // fprintf(fout, "# Directed RMAT Scale=%d Edgefactor=%d\n", scale, edgef);
  // fprintf(fout, "#\n");
  // fprintf(fout, "# Nodes: %"PRIu64" Edges: %"PRIu64"\n", max, myned);
  // fprintf(fout, "# NodeId\tNodeId\n");
  for (i = 0; i < myned; i++) {
    fprintf(fout, "%" PRIu64 "\t%" PRIu64 "\n", myedges[2 * i],
            myedges[2 * i + 1]);
  }
  fclose(fout);
}

/*
 * Graph Partitioning
 *
 * MPI exchange edges based on 2-D partitioning
 */
static uint64_t part_graph(int myid, int ntask, uint64_t **ed, uint64_t nedge,
                           int part_mode) {

  uint64_t i;

  uint64_t *s_ed = NULL;
  uint64_t *r_ed = *ed;

  uint64_t totrecv;

  uint64_t *soff = NULL;
  uint64_t *roff = NULL;
  uint64_t *send_n = NULL;
  uint64_t *recv_n = NULL;

  int *pmask = NULL;

  int n, p;
  MPI_Status *status;
  MPI_Request *request;

  /* compute processor mask for edges */
  pmask = (int *)Malloc(nedge * sizeof(*pmask));
  send_n = (uint64_t *)Malloc(ntask * sizeof(*send_n));
  for (i = 0; i < nedge; i++) {
    if (part_mode == 1) {
      pmask[i] = r_ed[2 * i] % ntask; // 1D partiorning

    } else
      pmask[i] = EDGE2PROC(r_ed[2 * i], r_ed[2 * i + 1]);
    send_n[pmask[i]]++;
  }

  /* sort edges by owner process (recv_n is used as a tmp) */
  soff = (uint64_t *)Malloc(ntask * sizeof(*soff));
  soff[0] = 0;
  for (p = 1; p < ntask; p++)
    soff[p] = soff[p - 1] + send_n[p - 1];

  recv_n = (uint64_t *)Malloc(ntask * sizeof(*recv_n));
  memcpy(recv_n, soff, ntask * sizeof(*soff));

  s_ed = (uint64_t *)Malloc(2 * nedge * sizeof(*s_ed));
  for (i = 0; i < nedge; i++) {
    s_ed[2 * recv_n[pmask[i]]] = r_ed[2 * i];
    s_ed[2 * recv_n[pmask[i]] + 1] = r_ed[2 * i + 1];
    recv_n[pmask[i]]++;
  }
  /* to proc k must be send send_n[k] edges starting at s_ei[soff[k]] */
  MPI_Alltoall(send_n, 1, MPI_UNSIGNED_LONG_LONG, recv_n, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_COMM_CLUSTER);
  if (send_n[myid] != recv_n[myid]) {
    fprintf(stderr, "[%d] Error in %s:%d\n", myid, __func__, __LINE__);
    exit(EXIT_FAILURE);
  }

  roff = (uint64_t *)Malloc(ntask * sizeof(*roff));
  roff[0] = 0;
  totrecv = recv_n[0];
  for (p = 1; p < ntask; p++) {
    totrecv += recv_n[p];
    roff[p] = roff[p - 1] + recv_n[p - 1];
  }
  r_ed = (uint64_t *)Realloc(r_ed, 2 * totrecv * sizeof(*r_ed));

  status = (MPI_Status *)Malloc(ntask * sizeof(*status));
  request = (MPI_Request *)Malloc(ntask * sizeof(*request));

  /* post RECVs */
  for (p = 0, n = 0; p < ntask; p++) {
    if (recv_n[p] == 0 || p == myid)
      continue;
    MPI_Irecv(r_ed + 2 * roff[p], 2 * recv_n[p], MPI_UNSIGNED_LONG_LONG, p,
              PTAG(p), MPI_COMM_CLUSTER, request + n);
    n++;
  }
  /* do the SENDs */
  memcpy(r_ed + 2 * roff[myid], s_ed + 2 * soff[myid],
         2 * send_n[myid] * sizeof(*s_ed));
  for (p = 0; p < ntask; p++) {
    if (send_n[p] == 0 || p == myid)
      continue;
    MPI_Send(s_ed + 2 * soff[p], 2 * send_n[p], MPI_UNSIGNED_LONG_LONG, p,
             PTAG(myid), MPI_COMM_CLUSTER);
  }
  MPI_Waitall(n, request, status);

  free(s_ed);
  free(send_n);
  free(soff);
  free(roff);
  free(recv_n);
  free(pmask);
  free(status);
  free(request);

  *ed = r_ed;
  return totrecv;
}

/*
 * Compare Edges
 *
 * Compares Edges p1 (a,b) with p2 (c,d) according to the following algorithm:
 * First compares the nodes where edges are assigned, if they are on the same
 * processor than compares
 * tail and than head
 */
static int cmpedge(const void *p1, const void *p2) {
  uint64_t *l1 = (uint64_t *)p1;
  uint64_t *l2 = (uint64_t *)p2;
  if (EDGE2PROC(l1[0], l1[1]) < EDGE2PROC(l2[0], l2[1]))
    return -1;
  if (EDGE2PROC(l1[0], l1[1]) > EDGE2PROC(l2[0], l2[1]))
    return 1;
  if (l1[0] < l2[0])
    return -1;
  if (l1[0] > l2[0])
    return 1;
  if (l1[1] < l2[1])
    return -1;
  if (l1[1] > l2[1])
    return 1;
  return 0;
}

static LOCINT degree_reduction(int myid, int ntask, uint64_t **ed,
                               uint64_t nedge, uint64_t **edrem, LOCINT *ner) {
  uint64_t u = -1, v = -1, next_u = -1, next_v = -1, prev_u = -1;
  uint64_t i, j;
  uint64_t *n_ed = NULL; // new edge list
  uint64_t *o_ed = NULL; // removed edge list
  uint64_t *r_ed = NULL; // input edge list

  uint64_t ncouple = 0;

  uint64_t nod = 0, ne = 0, pnod = 0, skipped = 0; // vrem=0;

  // Graph partitioning 1-D
  if (ntask > 1)
    nedge = part_graph(myid, ntask, ed, nedge, 1);

  // Sort Edges (u,v) by u
  qsort(*ed, nedge, sizeof(uint64_t[2]), cmpedge_1d);
  r_ed = *ed;

  // dump_edges(r_ed, nedge,"Degree Reduction Edges 1-D");
  n_ed = (uint64_t *)Malloc(2 * nedge * sizeof(*n_ed));
  o_ed = (uint64_t *)Malloc(2 * nedge * sizeof(*o_ed));

  fprintf(stdout, "[rank %d] Memory allocated\n", myid);

  for (i = 0; i < nedge - 1; i++) {
    u = r_ed[2 * i];
    v = r_ed[2 * i + 1]; // current is ( u,v )    next pair is next_u, next_t
                         // based on index j
    j = 2 * i + 2;
    next_u = r_ed[j];
    next_v = r_ed[j + 1];
    if ((u == v) || ((u == next_u) && (v == next_v))) { // Skip
      skipped++;
      prev_u = u;
      continue;
    }
    if ((u != next_u) && (u != prev_u)) {
      // This is a 1-degre remove
      o_ed[2 * nod] = v;
      o_ed[2 * nod + 1] = u;
      nod++;
    } else { // this is a first of a series or within a series
      n_ed[2 * ne] = u;
      n_ed[2 * ne + 1] = v;
      ne++;
    }
    prev_u = u;
  }
  // Check last edge
  u = r_ed[2 * nedge - 2];
  v = r_ed[2 * nedge - 1];
  if (u == prev_u) {
    n_ed[2 * ne] = u;
    n_ed[2 * ne + 1] = v;
    ne++;
  } else if (u != v) { // 1-degree store v before u
    o_ed[2 * nod] = v;
    o_ed[2 * nod + 1] = u;
    nod++;
  }

  fprintf(stdout, "[rank %d] Edges removed during fist step %lu\n", myid, nod);
  // 1-degree vertices (nod )removed
  // HERE n_ed contains the new edges list
  // o_ed contains removed edges list
  if (ntask > 1)
    nod = part_graph(myid, ntask, &o_ed, nod, 1); // partition removed edges

  // sort partitioned edges
  // Sort Edges (u,v) by u
  qsort(o_ed, nod, sizeof(uint64_t[2]), cmpedge_1d);

  // dump_edges(o_ed, nod, "edges removed");

  // remove edges for vertices removed in the previous step
  if (nod > 0) {
    // Number of edges left after 1-degree removal
    nedge = ne;
    // nod is the number of edges we need to remove after exchange
    // nedge are the number of edges we
    ne = 0;
    for (i = 0; i < nedge; i++) {
      // This is required to solve the case when two vertices are connected
      // between them but
      // disconnected from all the others
      while ((n_ed[2 * i] > o_ed[2 * pnod]) && (pnod < nod)) {
        ncouple++;
        pnod++;
      }

      if ((n_ed[2 * i] == o_ed[2 * pnod]) &&
          (n_ed[2 * i + 1] == o_ed[2 * pnod + 1])) {
        pnod++;
        // skip this
        continue;
      } else {
        // save this in the remaining edges
        r_ed[2 * ne] = n_ed[2 * i];
        r_ed[2 * ne + 1] = n_ed[2 * i + 1];
        ne++;
      }
    }
  } else
    memcpy(r_ed, n_ed, 2 * ne * sizeof(r_ed));

  fprintf(stdout, "[rank %d] Edges removed during second step %lu\n", myid,
          pnod);
  fprintf(stdout, "[rank %d] Couple of edges removed %lu\n", myid, ncouple);

  // dump_edges(r_ed,ne, "GRAPH FOR CSC");
  *ner = pnod;   // How many vertices have been removed
  *edrem = o_ed; // Array of removed edges
  // o_ed = NULL;  // ATTENZIONE !!
  free(n_ed);
  // free(o_ed);   //ATTENZIONE !! ???
  return ne;
}

/*
 *
 * ed   array with edges
 * ned  number of edges
 * deg  array with degrees
 *
 */

static uint64_t norm_graph(uint64_t *ed, uint64_t ned, LOCINT *deg) {

  uint64_t l, n;

  if (ned == 0)
    return 0;

  qsort(ed, ned, sizeof(uint64_t[2]), cmpedge);
  // record degrees considering multiple edges
  // and self-loop and remove them from edge list
  if (deg != NULL)
    deg[GI2LOCI(ed[0])]++;
  for (n = l = 1; n < ned; n++) {

    if (deg != NULL)
      deg[GI2LOCI(ed[0])]++;
    if (((ed[2 * n] !=
          ed[2 * (n - 1)]) || // Check if two consecutive heads are different
         (ed[2 * n + 1] != ed[2 * (n - 1) + 1])) && // Check if two consecutive
                                                    // tails are different
        (ed[2 * n] != ed[2 * n + 1])) {             // It is not a "cappio"

      ed[2 * l] = ed[2 * n]; // since it is not a "cappio" and is not a
                             // duplicate edge, copy it in the final edge array
      ed[2 * l + 1] = ed[2 * n + 1];
      l++;
    }
  }
  return l;
}

// probably unneeded
static int verify_32bit_fit(uint64_t *ed, uint64_t ned) {

  uint64_t i;

  for (i = 0; i < ned; i++) {
    uint64_t v;

    v = GI2LOCI(ed[2 * i]);
    if (v >> (sizeof(LOCINT) * 8)) {
      fprintf(stdout, "[%d] %" PRIu64 "=GI2LOCI(%" PRIu64
                      ") won't fit in a 32-bit word\n",
              myid, v, ed[2 * i]);
      return 0;
    }
    v = GJ2LOCJ(ed[2 * i + 1]);
    if (v >> (sizeof(LOCINT) * 8)) {
      fprintf(stdout, "[%d] %" PRIu64 "=GJ2LOCJ(%" PRIu64
                      ") won't fit in a 32-bit word\n",
              myid, v, ed[2 * i + 1]);
      return 0;
    }
  }
  return 1;
}

static void init_lcc_1degree(uint64_t *edrem, uint64_t nedrem, uint64_t nverts,
                             LOCINT *reach) {
  uint64_t i;
  LOCINT ur = 0;
  for (i = 0; i < nedrem; i++) {
    // Edrem are edges (u,v) where v is a 1-degree vertex removed
    ur = GI2LOCI(edrem[2 * i]); // this is local row
    // We need to use the number of vertices in the connected component
    reach[ur]++;
  }
}

/*
 * Build compressed sparse row
 *
 */

static void build_csc(uint64_t *ed, uint64_t ned, LOCINT **col, LOCINT **row) {

  LOCINT *r, *c, *tmp, i;

  /* count edges per col */
  tmp = (LOCINT *)Malloc(col_bl * sizeof(*tmp));
  for (i = 0; i < ned; i++)
    tmp[GJ2LOCJ(ed[2 * i + 1])]++; // Here we have the local degree (number of
                                   // edges for each local row)
  /* compute csc col[] vector with nnz in last element */
  c = (LOCINT *)Malloc((col_bl + 1) * sizeof(*c));
  c[0] = 0;
  for (i = 1; i <= col_bl; i++)
    c[i] = c[i - 1] + tmp[i - 1]; // Sum to the previous index the local degree.
  /* fill csc row[] vector */
  memcpy(tmp, c, col_bl * sizeof(*c)); /* no need to copy last int (nnz) */
  r = (LOCINT *)Malloc(ned * sizeof(*r));
  for (i = 0; i < ned; i++) {
    r[tmp[GJ2LOCJ(ed[2 * i + 1])]] = GI2LOCI(ed[2 * i]);
    tmp[GJ2LOCJ(ed[2 * i + 1])]++;
  }
  free(tmp);
  *row = r;
  *col = c;
  return;
}

/*
 * Compare unsigned local values
 */
static int cmpuloc(const void *p1, const void *p2) {

  LOCINT l1 = *(LOCINT *)p1;
  LOCINT l2 = *(LOCINT *)p2;
  if (l1 < l2)
    return -1;
  if (l1 > l2)
    return 1;
  return 0;
}

/*
 *
 *
 */
uint64_t compact(uint64_t *v, uint64_t ld, int *vnum, int n) {

  int i, j;
  uint64_t cnt = vnum[0];
  for (i = 1; i < n; i++)
    for (j = 0; j < vnum[i]; j++)
      v[cnt++] = v[i * ld + j];
  return cnt;
}

static inline void exchange_vert4x2(LOCINT *frt, LOCINT *frt_sig, int nfrt,
                                    LOCINT *rbuf, LOCINT ld, int *rnum,
                                    MPI_Request *request, MPI_Status *status,
                                    int post) {
  int i, p;
  ld = ld * 2;

  // Receive vertices from the processors in the same column except myself
  // There are R processors in the same column
  // Here I receive the frontiers from all other processors in the same column
  for (i = 1; i < R; i++) {
    p = (myrow + i) % R;
    MPI_Irecv(rbuf + p * ld, ld, LOCINT_MPI, pmesh[p][mycol],
              VTAG(pmesh[p][mycol]), MPI_COMM_CLUSTER, request + i - 1);
  }
  // Copy in the receiving buffer my frontier
  memcpy(rbuf + myrow * ld, frt, nfrt * sizeof(*frt));
  // Copy in the receiving buffer sigma values right after the frontier
  memcpy(rbuf + myrow * ld + nfrt, frt_sig, nfrt * sizeof(*frt_sig));
  // Store in rnum number of vertices in the new frontier found on this
  // processor
  rnum[myrow] = nfrt;
  // Send the frontier to all processors in the same Column.
  // Here we send all vertices in the froniter to all processors on the column
  for (i = 1; i < R; i++) {
    p = (myrow + i) % R;
    MPI_Send(rbuf + myrow * ld, 2 * nfrt, LOCINT_MPI, pmesh[p][mycol],
             VTAG(myid), MPI_COMM_CLUSTER);
  }
  // Wait for IRecv to complete
  MPI_Waitall(R - 1, request, status);
  for (i = 1; i < R; i++) {
    // Get how many vertices have been received from each processor and store
    // the value in rnum[]
    p = (myrow + i) % R;
    MPI_Get_count(status + i - 1, LOCINT_MPI, rnum + p);
    // Receive both vertices and their sigma value
    if (rnum[p] > 0)
      rnum[p] =
          rnum[p] / 2; // In this way rnum contains the number of couples V,S
  }
  return;
}

static inline void exchange_horiz4x2(LOCINT *sbuf, LOCINT sld, int *snum,
                                     LOCINT *rbuf, LOCINT rld, int *rnum,
                                     MPI_Request *request, MPI_Status *status,
                                     int post) {
  int i, p;

  rld = rld * 2;
  sld = sld * 2;

  // Post the IRecv for all processes in the same row
  for (i = 1; i < C; i++) {
    p = (mycol + i) % C;
    MPI_Irecv(rbuf + p * rld, rld, LOCINT_MPI, pmesh[myrow][p],
              HTAG(pmesh[myrow][p]), MPI_COMM_CLUSTER, request + i - 1);
  }

  rnum[mycol] = 0;

  for (i = 1; i < C; i++) {
    // Send data to other processes
    p = (mycol + i) % C;
    MPI_Send(sbuf + p * sld, snum[p] * 2, LOCINT_MPI, pmesh[myrow][p],
             HTAG(myid), MPI_COMM_CLUSTER);
  }
  MPI_Waitall(C - 1, request, status);
  for (i = 1; i < C; i++) {
    // Get the real number of data sent
    MPI_Get_count(status + i - 1, LOCINT_MPI, rnum + (mycol + i) % C);
    if (rnum[(mycol + i) % C] > 0)
      rnum[(mycol + i) % C] = rnum[(mycol + i) % C] / 2;
  }

  return;
}

static void dump_deg(LOCINT *deg, LOCINT *deg_count, int n) {

  FILE *fp = NULL;
  char name[MAX_LINE];
  int myid;
  int i;
  MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);
  snprintf(name, MAX_LINE, "degree_%d", myid);
  fp = Fopen(name, "a");

  for (i = 0; i < n; i++)
    fprintf(fp, " %d (%d),", deg[i], deg_count[i]);

  fprintf(fp, "\n");
  fclose(fp);
  return;
}

static void dump_array(const char *name, LOCINT *arr, int n) {

  FILE *fp = NULL;
  char fname[MAX_LINE];
  int myid;
  int i;

  MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);
  snprintf(fname, MAX_LINE, "%s_%d", name, myid);
  fp = Fopen(fname, "a");

  for (i = 0; i < n; i++)
    fprintf(fp, " %d,", arr[i]);

  fprintf(fp, "\n");
  fclose(fp);
  return;
}

static void analyze_deg(LOCINT *deg, int n) {

  int i, curr_index = 0, new_count;
  char fname[256];

  LOCINT *deg_unique = (LOCINT *)Malloc(col_bl * sizeof(*deg_unique));
  LOCINT *deg_count = (LOCINT *)Malloc(col_bl * sizeof(*deg_count));

  memcpy(deg_unique, deg, col_bl * sizeof(*deg_unique));
  qsort(deg_unique, n, sizeof(LOCINT), cmpuloc);

  deg_count[curr_index] = 1;

  for (i = 1; i < n; i++) {
    if (deg_unique[i] == deg_unique[curr_index]) {
      deg_count[curr_index]++;
    } else {
      curr_index++;
      deg_unique[curr_index] = deg_unique[i];
      deg_count[curr_index] = 1;
    }
  }
  new_count = curr_index + 1;

  int bin_count = 16;
  LOCINT bin_limits[16] = {0, 1, 2,  3,   4,    5,     6,      7,
                           8, 9, 10, 100, 1000, 10000, 100000, 10000000};
  // int bin_count = sizeof(bin_limits);

  LOCINT *bins = NULL;
  bins = (LOCINT *)Malloc(bin_count * sizeof(*bins));

  memset(bins, 0, bin_count * sizeof(*bins));

  i = new_count - 1;
  int curr_bin = (bin_count - 1);

  while (i > -1) {
    if (deg_unique[i] >= bin_limits[curr_bin]) {
      bins[curr_bin] += deg_count[i];
      i--;
    } else {
      curr_bin--;
      if (curr_bin < 0)
        break;
    }
  }

  // dump_deg(deg_unique, deg_count, new_count);

  snprintf(fname, 256, "degree_stats_%d", outId);
  dump_array(fname, bins, bin_count);

  freeMem(deg_unique);
  freeMem(deg_count);
}

enum {
  s_minimum,
  s_firstquartile,
  s_median,
  s_thirdquartile,
  s_maximum,
  s_mean,
  s_std,
  s_LAST
};

static int compare_doubles(const void *a, const void *b) {

  double aa = *((const double *)a);
  double bb = *((const double *)b);

  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

static void get_statistics(const double x[], int n, double r[s_LAST]) {

  double temp;
  int i;

  /* Compute mean. */
  temp = 0;
  for (i = 0; i < n; ++i)
    temp += x[i];
  temp /= n;
  r[s_mean] = temp;

  /* Compute std. dev. */
  temp = 0;
  for (i = 0; i < n; ++i)
    temp += (x[i] - r[s_mean]) * (x[i] - r[s_mean]);
  temp /= n - 1;
  r[s_std] = sqrt(temp);
  r[s_std] /= (r[s_mean] * r[s_mean] * sqrt(n - 1));

  /* Sort x. */
  double *xx = (double *)Malloc(n * sizeof(double));
  memcpy(xx, x, n * sizeof(double));
  qsort(xx, n, sizeof(double), compare_doubles);

  /* Get order statistics. */
  r[s_minimum] = xx[0];
  r[s_firstquartile] = (xx[(n - 1) / 4] + xx[n / 4]) * .5;
  r[s_median] = (xx[(n - 1) / 2] + xx[n / 2]) * .5;
  r[s_thirdquartile] = (xx[n - 1 - (n - 1) / 4] + xx[n - 1 - n / 4]) * .5;
  r[s_maximum] = xx[n - 1];

  /* Clean up. */
  free(xx);
}

static void print_stats(double *teps, int n) {

  int i;
  double stats[s_LAST];

  for (i = 0; i < n; i++)
    teps[i] = 1.0 / teps[i];

  get_statistics(teps, n, stats);

  fprintf(stdout, "TEPS statistics:\n");
  fprintf(stdout, "\t   harm mean: %lf\n", 1.0 / stats[s_mean]);
  fprintf(stdout, "\t   harm stdev: %lf\n", stats[s_std]);
  fprintf(stdout, "\t   median: %lf\n", 1.0 / stats[s_median]);
  fprintf(stdout, "\t   minimum: %lf\n", 1.0 / stats[s_maximum]);
  fprintf(stdout, "\t   maximum: %lf\n", 1.0 / stats[s_minimum]);
  fprintf(stdout, "\tfirstquartile: %lf\n", 1.0 / stats[s_firstquartile]);
  fprintf(stdout, "\tthirdquartile: %lf\n", 1.0 / stats[s_thirdquartile]);
  return;
}

void usage(const char *pname) {

  prexit("Usage:\n"
         "\t %1$s -p RxC [-d dev0,dev1,...] [-o outfile] [-D] [-d] [-m] [-N <# "
         "of searches>]\n"
         "\t -> to visit a graph read from file:\n"
         "\t\t -f <graph file> -n <# vertices>\n"
         "\t\t -S <scale> [-E <edge factor>]\n"
         "\t Where:\n"
         "\t\t -D to ENABLE debug information\n"
         "\t\t -U DO NOT make graph Undirected\n"
         "\t\t -a perform degree analysis\n"
         "\n",
         pname);
  return;
}


LOCINT bin_search(LOCINT *arr, int l, int r, LOCINT x){
	while (l <= r) {
		int m = l + (r - l) / 2;
		if (arr[m] == x) return 1;
		if (arr[m] < x) l = m + 1;
		else r = m - 1;
	}
}


#ifdef HAVE_SIMD
void lcc_func_bin_simd(LOCINT *col, LOCINT *row, float *output) {
  LOCINT i = 0;
  int dest_get;
  LOCINT jj = 0;
  LOCINT vv = 0;
  LOCINT vvv = 0;
  LOCINT counter = 0;
  LOCINT gvid = 0;
  LOCINT r_off[2] = {0, 0};
  LOCINT row_offset, off_start = 0;
  float lcc = 0;
// Variable for SIMD 
  LOCINT c = 0;
  static LOCINT local_counter = 0;
  static int tid;
  LOCINT reduction[32];


#ifdef HAVE_CLAMPI
  CMPI_Win win_col;
  CMPI_Win win_row;
  CMPI_Win_create(col, col_bl * sizeof(LOCINT), sizeof(LOCINT), MPI_INFO_NULL,
                    Row_comm, &win_col);
  CMPI_Win_create(row, col[col_bl] * sizeof(LOCINT), sizeof(LOCINT),
                    MPI_INFO_NULL, Row_comm, &win_row);
  MMPI_WIN_LOCK_ALL(0, win_col.win);
  MMPI_WIN_LOCK_ALL(0, win_row.win);
#else
  MMPI_WIN win_col;
  MMPI_WIN win_row;
  MMPI_WIN_CREATE(col, col_bl * sizeof(LOCINT), sizeof(LOCINT), MPI_INFO_NULL,
                    Row_comm, &win_col);
  MMPI_WIN_CREATE(row, col[col_bl] * sizeof(LOCINT), sizeof(LOCINT),
                    MPI_INFO_NULL, Row_comm, &win_row);
  MMPI_WIN_LOCK_ALL(0, win_col);
  MMPI_WIN_LOCK_ALL(0, win_row);
#endif

  LOCINT *adj_v = (LOCINT *)Malloc(row_pp * sizeof(LOCINT));
  LOCINT *adj_local = (LOCINT *)Malloc(row_pp * sizeof(LOCINT));

  LOCINT nget = 0;
  LOCINT nlocal = 0;
  LOCINT zerouno = 0;
#ifdef HAVE_LIBLSB
  LSB_Set_Rparam_int("rank", gmyid);
  LSB_Set_Rparam_int("csize", gntask);

#ifdef HAVE_CLAMPI
  LSB_Set_Rparam_string("type", "CLAMPI");
#else
  LSB_Set_Rparam_string("type", "MPI");
#endif

#endif

  int gres;
  int it;

  for (it = 0; it < RUNS + WARMUP; it++) {
#ifdef HAVE_LIBLSB
      LSB_Res();
#endif

#pragma omp threadprivate(local_counter, tid)
#pragma omp parallel
      tid = omp_get_thread_num();
      for (i = 0; i < col_bl; i++) {
          row_offset = col[i + 1] - col[i];
          memcpy(adj_local, &row[col[i]], row_offset * sizeof(LOCINT));
          counter = 0;
#pragma omp parallel
{
          local_counter = 0;
}

// USE A CONSTANT HERE DEPENDS ON HOW MANY COREs YOU ARE USING 
          for (c = 0; c < 32; c++) reduction[c] = 0;

          for (jj = 0; jj < row_offset; jj++) {
              gvid = LOCI2GI(row[col[i] + jj]); // offset gvid in proc dest_get is gvid % C
              dest_get = VERT2PROC(gvid);
              off_start = gvid % col_bl;
              // continue; skip degree-1 to do
              if (dest_get != myid) {
#ifdef HAVE_CLAMPI
                  gres = CMPI_Get(r_off, 2, MPI_UINT32_T, dest_get, off_start, 2, MPI_UINT32_T, win_col);
                  if (gres != CL_HIT) CMPI_Win_flush(dest_get, win_col);
#else
                  MMPI_GET(r_off, 2, MPI_UINT32_T, dest_get, off_start, 2, MPI_UINT32_T, win_col);
                  MMPI_WIN_FLUSH(dest_get, win_col);
#endif

#ifdef HAVE_CLAMPI
                  gres =
                     CMPI_Get(adj_v, r_off[1] - r_off[0], MPI_UINT32_T, dest_get,
                         r_off[0], r_off[1] - r_off[0], MPI_UINT32_T, win_row);
                  if (gres != CL_HIT) CMPI_Win_flush(dest_get, win_row);
#else
                  MMPI_GET(adj_v, r_off[1] - r_off[0], MPI_UINT32_T, dest_get,
                     r_off[0], r_off[1] - r_off[0], MPI_UINT32_T, win_row);
                  MMPI_WIN_FLUSH(dest_get, win_row);
#endif
                  nget++;
             } 
	     else {
                  r_off[0] = col[off_start];
                  r_off[1] = col[off_start + 1];
                  memcpy(adj_v, &row[r_off[0]], (r_off[1] - r_off[0]) * sizeof(LOCINT));
                  nlocal++;
             }
             // Compute LCC
             int r_offset = r_off[1]-r_off[0];
             if (r_offset < 32){
	          for (vv = 0; vv < r_offset; vv++){
		      if (adj_v[vv] == LOCI2GI(i)) continue;
		      if (bin_search(adj_local,0, row_offset-1, adj_v[vv]))
			      counter +=1;
	          }
             }
             else{
#pragma omp parallel for schedule(dynamic, 8) // Put a define here 
	          for (vv = 0; vv < r_offset; vv++){
		      if (adj_v[vv] == LOCI2GI(i)) continue;
		      if (bin_search(adj_local,0, row_offset-1, adj_v[vv])) local_counter += 1;
		      reduction[tid] = local_counter;
	          }
             }
	  } //end jj loop
          if (row_offset == 1 || row_offset == 0){
		  lcc = 0;
	  }  
	  else{  // Put the define here   
                  for (c = 0; c < 32; c++) counter += reduction[c];
                  lcc = (float) counter/(float)(row_offset*(row_offset-1));
	  }
          output[i] = lcc;
#ifdef HAVE_LIBLSB
      if (it > WARMUP)
        LSB_Rec(it);
#endif
      }// end col_block loop
#ifdef HAVE_CLAMPI
      CMPI_Win_invalidate(win_col);
      CMPI_Win_invalidate(win_row);
#endif
    }

#ifdef HAVE_CLAMPI
    MMPI_WIN_UNLOCK_ALL(win_col.win);
    MMPI_WIN_UNLOCK_ALL(win_row.win);
    CMPI_Win_invalidate(win_col);
    CMPI_Win_invalidate(win_row);

    CMPI_Win_free(&win_col);
    CMPI_Win_free(&win_row);
#else
    MMPI_WIN_UNLOCK_ALL(win_col);
    MMPI_WIN_UNLOCK_ALL(win_row);
    MMPI_WIN_FREE(&win_col);
    MMPI_WIN_FREE(&win_row);
#endif

    freeMem(adj_v);
    freeMem(adj_local);
}
#endif




void lcc_func(LOCINT *col, LOCINT *row, float *output) {
  LOCINT i = 0;
  if (R == 1) {
    int dest_get;
    LOCINT jj = 0;
    LOCINT vv = 0;
    LOCINT vvv = 0;
    LOCINT counter = 0;
    LOCINT gvid = 0;
    LOCINT r_off[2] = {0, 0};
    LOCINT row_offset, off_start = 0;
    float lcc = 0;

#ifdef HAVE_CLAMPI
    CMPI_Win win_col;
    CMPI_Win win_row;
    CMPI_Win_create(col, col_bl * sizeof(LOCINT), sizeof(LOCINT), MPI_INFO_NULL,
                    Row_comm, &win_col);
    CMPI_Win_create(row, col[col_bl] * sizeof(LOCINT), sizeof(LOCINT),
                    MPI_INFO_NULL, Row_comm, &win_row);
    MMPI_WIN_LOCK_ALL(0, win_col.win);
    MMPI_WIN_LOCK_ALL(0, win_row.win);
#else
    MMPI_WIN win_col;
    MMPI_WIN win_row;
    MMPI_WIN_CREATE(col, col_bl * sizeof(LOCINT), sizeof(LOCINT), MPI_INFO_NULL,
                    Row_comm, &win_col);
    MMPI_WIN_CREATE(row, col[col_bl] * sizeof(LOCINT), sizeof(LOCINT),
                    MPI_INFO_NULL, Row_comm, &win_row);
    MMPI_WIN_LOCK_ALL(0, win_col);
    MMPI_WIN_LOCK_ALL(0, win_row);
#endif

    LOCINT *adj_v = (LOCINT *)Malloc(row_pp * sizeof(LOCINT));
    LOCINT *adj_local = (LOCINT *)Malloc(row_pp * sizeof(LOCINT));

    LOCINT nget = 0;
    LOCINT nlocal = 0;
    LOCINT zerouno = 0;
#ifdef HAVE_LIBLSB
    LSB_Set_Rparam_int("rank", gmyid);
    LSB_Set_Rparam_int("csize", gntask);

#ifdef HAVE_CLAMPI
    LSB_Set_Rparam_string("type", "CLAMPI");
#else
    LSB_Set_Rparam_string("type", "MPI");
#endif

#endif

    int gres;
    int it;

    for (it = 0; it < RUNS + WARMUP; it++){
#ifdef HAVE_LIBLSB
      LSB_Res();
#endif
      for (i = 0; i < col_bl; i++) {

        row_offset = col[i + 1] - col[i];
        memcpy(adj_local, &row[col[i]], row_offset * sizeof(LOCINT));
        counter = 0;
        for (jj = 0; jj < row_offset; jj++) {
          gvid = LOCI2GI(row[col[i] + jj]); // offset gvid in proc dest_get is gvid % C
          dest_get = VERT2PROC(gvid);
          off_start = gvid % col_bl;
          // continue;
          if (dest_get != myid) {
#ifdef HAVE_CLAMPI
            gres = CMPI_Get(r_off, 2, MPI_UINT32_T, dest_get, off_start, 2,
                            MPI_UINT32_T, win_col);
            if (gres != CL_HIT)
              CMPI_Win_flush(dest_get, win_col);
#else
            MMPI_GET(r_off, 2, MPI_UINT32_T, dest_get, off_start, 2,
                     MPI_UINT32_T, win_col);
            MMPI_WIN_FLUSH(dest_get, win_col);
#endif

#ifdef HAVE_CLAMPI
            gres =
                CMPI_Get(adj_v, r_off[1] - r_off[0], MPI_UINT32_T, dest_get,
                         r_off[0], r_off[1] - r_off[0], MPI_UINT32_T, win_row);
            if (gres != CL_HIT)
              CMPI_Win_flush(dest_get, win_row);
#else
            MMPI_GET(adj_v, r_off[1] - r_off[0], MPI_UINT32_T, dest_get,
                     r_off[0], r_off[1] - r_off[0], MPI_UINT32_T, win_row);
            MMPI_WIN_FLUSH(dest_get, win_row);
#endif

            nget++;
          } else {
            r_off[0] = col[off_start];
            r_off[1] = col[off_start + 1];
            memcpy(adj_v, &row[r_off[0]],
                   (r_off[1] - r_off[0]) * sizeof(LOCINT));
            nlocal++;
          }
          // Compute LCC
          for (vv = 0; vv < r_off[1] - r_off[0]; vv++) {
            if (adj_v[vv] == LOCI2GI(i))
              continue;
            for (vvv = 0; vvv < row_offset; vvv++) {
              if (adj_v[vv] == adj_local[vvv]) {
                counter += 1;
              }
            }
          }
        }
        if (row_offset == 1 || row_offset == 0) {
          lcc = 0;
          zerouno++;
        } else {
          lcc = (float)counter / (float)(row_offset * (row_offset - 1));
        }

        output[i] = lcc;
      }
#ifdef HAVE_LIBLSB
      if (it > WARMUP)
        LSB_Rec(it);
#endif

#ifdef HAVE_CLAMPI
      CMPI_Win_invalidate(win_col);
      CMPI_Win_invalidate(win_row);
      //cl_flush(win_col);
      //cl_flush(win_row);
#endif
    }

    // fprintf(stdout,"%d %u %u %u\n", myid, col_bl, nget, nlocal);

#ifdef HAVE_CLAMPI
    MMPI_WIN_UNLOCK_ALL(win_col.win);
    MMPI_WIN_UNLOCK_ALL(win_row.win);
    //cl_flush(win_col);
    //cl_flush(win_row);
    CMPI_Win_invalidate(win_col);
    CMPI_Win_invalidate(win_row);

    CMPI_Win_free(&win_col);
    CMPI_Win_free(&win_row);
#else
    MMPI_WIN_UNLOCK_ALL(win_col);
    MMPI_WIN_UNLOCK_ALL(win_row);
    MMPI_WIN_FREE(&win_col);
    MMPI_WIN_FREE(&win_row);
#endif

    freeMem(adj_v);
    freeMem(adj_local);
  }
}

void init_rand(int mode) {
  uint64_t seed = DEFAULT_SEED;
  if (mode == 1) {
    struct timeval time;
    gettimeofday(&time, NULL);
    seed = getpid() + time.tv_sec + time.tv_usec;
  }
  srand48(seed);
}

LOCINT uniform_int(LOCINT max, LOCINT min) {
  LOCINT temp = lrand48() % (max - min + 1) + min;
  MPI_Bcast(&temp, 1, MPI_INT, myid, MPI_COMM_CLUSTER);
  return temp;
}

double uniform_double(double max, double min) {
  double temp = (double)rand() / RAND_MAX;
  temp = min + temp * (max - min + 1);
  return temp;
}

double uniform01() {
  double temp = (double)lrand48() / (double)RAND_MAX;
  MPI_Bcast(&temp, 1, MPI_DOUBLE, myid, MPI_COMM_CLUSTER);
  return temp;
}

int main(int argc, char *argv[]) {

  int s, t, gread = -1;
  int scale = 21, edgef = 16;
  short dump = 0, debug = 0;
  int64_t i, j;
  uint64_t nbfs = 1;
  LOCINT n, l, ned, rem_ed = 0;

  uint64_t *edge = NULL;
  uint64_t *rem_edge = NULL;

  LOCINT *col = NULL;
  LOCINT *row = NULL;

  LOCINT *degree = NULL; // Degree for all vertices in the same column

  float *hRFbuf = NULL;
  float *hSFbuf = NULL;
  LOCINT *deg = NULL;

  LOCINT *vRbuf = NULL;
  int *vRnum = NULL; /* type int (MPI_Send()/Recv() assumes int counts) */

  LOCINT *hSbuf = NULL;
  LOCINT *hRbuf = NULL;
  int *hSnum = NULL;
  int *hRnum = NULL;
  MPI_Status *status;
  MPI_Request *vrequest;
  MPI_Request *hrequest;
  MPI_Comm MPI_COMM_COL;

  int rootset = 0;

  int cntask;
  char *gfile = NULL, *p = NULL, c, *ofile = NULL;

  TIMER_DEF(0);

  int random = 0;
  /*Variables for approximating LCC/Triangle Counting*/
  LOCINT approx_nverts =
      1; // How many vertices to check the approximation score
  LOCINT *approx_vertices = NULL;

  float *dist_lcc = NULL; // Local Cluster Coef... Store in scan mode

  MMPI_INIT(&argc, &argv);
  MPI_Barrier(MPI_COMM_WORLD);
#ifdef HAVE_LIBLSB
  LSB_Init("llc", 0);
#endif

#ifdef HAVE_CLAMPI
  cl_init();
#endif

  MPI_Comm_rank(MPI_COMM_WORLD, &gmyid);
  MPI_Comm_size(MPI_COMM_WORLD, &gntask);

  if (argc == 1) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  outId = time(NULL);

  int wr, pl = 0, limit = sizeof(cmdLine);
  wr = snprintf(cmdLine + pl, limit, "MPI Tasks %d\n", gntask);
  if (wr < 0)
    exit(EXIT_FAILURE);
  limit -= wr;
  pl += wr;
  for (i = 0; i < argc; i++) {
    wr = snprintf(cmdLine + pl, limit, " %s", argv[i]);
    if (wr < 0)
      exit(EXIT_FAILURE);
    limit -= wr;
    pl += wr;
  }
  snprintf(cmdLine + pl, limit, "\n");

  if (gmyid == 0) {
    fprintf(stdout, "%s\n", cmdLine);
  }

  while ((c = getopt(argc, argv, "o:p:ahD:dUf:n:r:S:E:N:H:")) != EOF) {
#define CHECKRTYPE(exitval, opt)                                               \
  {                                                                            \
    if (exitval == gread)                                                      \
      prexit("Unexpected option -%c!\n", opt);                                 \
    else                                                                       \
      gread = !exitval;                                                        \
  }
    switch (c) {
    // BC approx  c param is the costanst used in Bader stopping cretierion
    case 'H':
      if (0 == sscanf(optarg, "%d", &heuristic))
        prexit("Invalid Heuristic Option (-H): %s\n", optarg);
      if (heuristic >= 2)
        prexit("H2 and H3 heuristics are not implemented(-H): %s\n", optarg);
      break;
    // heuristic selection
    case 'o':
      ofile = strdup(optarg);
      break;
    case 'p':
      strncpy(strmesh, optarg, 10);
      p = strtok(optarg, "x");
      if (!p)
        prexit("Invalid proc mesh field.\n");
      if (0 == sscanf(p, "%d", &R))
        prexit("Invalid number of rows for proc mesh (-p): %s\n", p);
      p = strtok(NULL, "x");
      if (!p)
        prexit("Invalid proc mesh field.\n");
      if (0 == sscanf(p, "%d", &C))
        prexit("Invalid number of columns for proc mesh (-p): %s\n", p);
      break;
    case 'd':
      // Dump RMAT Generated graph
      dump = 1;
      break;
    case 'D':
      // DEBUG
      debug = 1;
      break;
    case 'f':
      CHECKRTYPE(0, 'f')
      gfile = strdup(optarg);
      break;
    case 'n':
      CHECKRTYPE(0, 'n')
      if (0 == sscanf(optarg, "%" PRIu64, &N))
        prexit("Invalid number of vertices (-n): %s\n", optarg);
      break;
    case 'S':
      CHECKRTYPE(1, 'S')
      if (0 == sscanf(optarg, "%d", &scale))
        prexit("Invalid scale (-S): %s\n", optarg);
      N = ((uint64_t)1) << scale;
      break;
    case 'E':
      CHECKRTYPE(1, 'E')
      if (0 == sscanf(optarg, "%d", &edgef))
        prexit("Invalid edge factor (-S): %s\n", optarg);
      break;
    case 'N':
      if (0 == sscanf(optarg, "%ld", &nbfs))
        prexit("Option not implemented (-N): %s\n", optarg);
      break;
    case 'U':
      // Undirected
      undirected = 0;
      break;
    case 'a':
      // Degree analysis
      analyze_degree = 1;
      break;
    case 'h':
    case '?':
    default:
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
#undef CHECKRTYPE
  }
  if (approx_nverts > N)
    prexit("Number of vertices required is lager than N\n");
  if (gread) {
    if (!gfile || !N)
      prexit("Graph file (-f) and number of vertices (-n)"
             " must be specified for file based bfs.\n");
  }

  if (0 >= R || MAX_PROC_I < R || 0 >= C || MAX_PROC_J < C)
    prexit("R and C must be in range [1,%d] and [1,%d], respectively.\n",
           MAX_PROC_I, MAX_PROC_J);

  if (0 != N % (R * C))
    prexit("N must be multiple of both R and C.\n");

  ntask = R * C;
  cntask = gntask / ntask;
  if (gntask % ntask) {
    fprintf(stderr, "Invalid configuration: total number of task is %d, "
                    "cluster size is %d\n",
            gntask, ntask);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (heuristic >= 2) {
    prexit("\n\nH2 and H3 are not implemented(-H): %d\n", heuristic);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

#ifndef _LARGE_LVERTS_NUM
  if ((N / R) > UINT32_MAX) {
    prexit("Number of vertics per processor too big (%" LOCPRI "), please"
           "define _LARGE_LVERTS_NUM macro in %s.\n",
           (N / R), __FILE__);
  }
#endif

  int color = gmyid / ntask;
  MPI_Comm_split(MPI_COMM_WORLD, color, gmyid, &MPI_COMM_CLUSTER);
  MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);
  MPI_Comm_size(MPI_COMM_CLUSTER, &ntask);

  myrow = myid / C;
  mycol = myid % C;

  MPI_Comm_split(MPI_COMM_WORLD, (myrow * C) + mycol, gmyid, &MPI_COMM_COL);

  row_bl = N / (R * C); /* adjacency matrix rows per block:    N/(RC) */
  col_bl = N / C;       /* adjacency matrix columns per block: N/C */
  row_pp = N / R;       /* adjacency matrix rows per proc:     N/(RC)*C = N/R */

  if ((gmyid == 0) && (debug == 1)) {
    char fname[MAX_LINE];
    snprintf(fname, MAX_LINE, "%s_%d.log", "debug", gmyid);
    outdebug = Fopen(fname, "w");
  }

  char *resname = NULL;

  if (ntask > 1)
    mono = 0;
  // Disable random when a starting node is provided
  if (rootset > 0)
    random = 0;

  if (gmyid == 0) {
    fprintf(stdout, "\n\n****** DEVEL VERSION "
                    "******\n\n\n\n***************************\n\n");
    fprintf(stdout, "Total number of vertices (N): %" PRIu64 "\n", N);
    fprintf(stdout, "Processor mesh rows (R): %d\n", R);
    fprintf(stdout, "Processor mesh columns (C): %d\n", C);
    if (gread) {
      fprintf(stdout, "Reading graph from file: %s\n", gfile);
    } else {
      fprintf(stdout, "RMAT graph scale: %d\n", scale);
      fprintf(stdout, "RMAT graph edge factor: %d\n", edgef);
      fprintf(stdout, "Number of bc rounds: %ld\n", nbfs);
    }
    fprintf(stdout, "\n\n");
    if (heuristic == 0) {
      fprintf(stdout, "HEURISTICs: OFF: %d\n", heuristic);

    } else if (heuristic == 1) {
      fprintf(stdout, "HEURISTICs: 1-degree reduction ON: %d\n", heuristic);
    }

    fprintf(stdout, "\n");
  }

  if (NULL != ofile) {
    fprintf(stdout, "Result written to file: %s\n", ofile);
    resname = (char *)malloc((sizeof(ofile) + MAX_LINE) * sizeof(*resname));
    sprintf(resname, "%s_%dX%d_%d.log", ofile, R, C, gmyid);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* fill processor mesh */
  memset(pmesh, -1, sizeof(pmesh));
  for (i = 0; i < R; i++)
    for (j = 0; j < C; j++)
      pmesh[i][j] = i * C + j;

  if (myid == 0)
    fprintf(stdout, "%s graph...\n", gread ? "Reading" : "Generating");
  TIMER_START(0);
  if (gread)
    ned = read_graph(myid, ntask, gfile, &edge); // Read from file
  else
    ned = gen_graph(scale, edgef, &edge); // Generate RMAT
  TIMER_STOP(0);
  if (myid == 0)
    fprintf(stdout, " done in %f secs\n", TIMER_ELAPSED(0) / 1.0E+6);
  prstat(ned, gread ? "Edges read from file:" : "Edges generated:", 1);

  if (heuristic != 0) {
    l = norm_graph(edge, ned, NULL);
    prstat(ned - l, "First Multi-edges removed:", 1);
    ned = l;
  }

  if (dump > 0 && gread == 0 && ntask == 1) {
    fprintf(stdout, "Dump file...\n");
    dump_rmat(edge, ned, scale, edgef);
  }

  // 1 DEGREE PREPROCESSING TIMING ON
  if (heuristic == 1) {
    if (gmyid == 0)
      fprintf(stdout, "Degree reduction graph (%d)...\n", heuristic);
    TIMER_START(0);
    // DEGREE REDUCTION - Edge Based
    ned = degree_reduction(myid, ntask, &edge, ned, &rem_edge, &rem_ed);
    TIMER_STOP(0);
    degree_reduction_time = TIMER_ELAPSED(0);
  }
  // 2-D PARTITIONING
  if (gmyid == 0)
    fprintf(stdout, "Partitioning graph... ");
  TIMER_START(0);
  if (ntask > 1)
    ned = part_graph(myid, ntask, &edge, ned, 2); // 2-D graph partitioning
  if (ntask > 1)
    rem_ed = part_graph(myid, ntask, &rem_edge, rem_ed, 2);
  TIMER_STOP(0);
  if (myid == 0)
    fprintf(stdout, "task %d done in %f secs\n", gmyid,
            TIMER_ELAPSED(0) / 1.0E+6);
  prstat(ned, "Edges assigned after partitioning:", 1);
// dump_edges(edge, ned,"Edges 2-D");
#ifndef _LARGE_LVERTS_NUM
  if (ned > UINT32_MAX) {
    fprintf(stderr, "Too many vertices assigned to me. Change LOCINT\n");
    exit(EXIT_FAILURE);
  }
#endif
  if (myid == 0)
    fprintf(stdout, "task %d: Verifying partitioning...", gmyid);
  TIMER_START(0);
  for (n = 0; n < ned; n++) {
    if (EDGE2PROC(edge[2 * n], edge[2 * n + 1]) != myid) {
      fprintf(stdout, "[%d] error, received edge (%" PRIu64 ", %" PRIu64
                      "), should have been sent to %d\n",
              myid, edge[2 * n], edge[2 * n + 1],
              EDGE2PROC(edge[2 * n], edge[2 * n + 1]));
      break;
    }
  }
  s = (n != ned);
  MPI_Allreduce(&s, &t, 1, MPI_INT, MPI_LOR, MPI_COMM_CLUSTER);
  TIMER_STOP(0);
  if (t)
    prexit("Error in 2D decomposition.\n");
  else if (myid == 0)
    fprintf(stdout, "task %d done in %f secs\n", gmyid,
            TIMER_ELAPSED(0) / 1.0E+6);

  /* Graph has been partitioned correctly */
  if (myid == 0)
    fprintf(stdout, "task %d Removing multi-edges...", gmyid);
  deg = (LOCINT *)Malloc(row_pp * sizeof(*deg));

  TIMER_START(0);
  // Normalize graph: remove loops and duplicates edges cappi o loops?
  // THIS ALSO CALCULATES DEGREES
  l = norm_graph(edge, ned, deg);
  TIMER_STOP(0);
  if (myid == 0)
    fprintf(stdout, "task %d done in %f secs\n", gmyid,
            TIMER_ELAPSED(0) / 1.0E+6);
  prstat(ned - l, "Multi-edges removed:", 1);
  ned = l;

  // check whether uint64 edges can fit in 32bit CSC
  if (4 == sizeof(LOCINT)) {
    if (!verify_32bit_fit(edge, ned))
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (myid == 0)
    fprintf(stdout, "task %d, Creating CSC...", gmyid);
  TIMER_START(0);

  build_csc(edge, ned, &col, &row); // Make the CSC Structure
  TIMER_STOP(0);
  if (myid == 0)
    fprintf(stdout, "task %d done in %f secs\n", gmyid,
            TIMER_ELAPSED(0) / 1.0E+6);
  freeMem(edge);
  freeMem(deg);

  MPI_Comm_split(MPI_COMM_CLUSTER, myrow, mycol, &Row_comm);
  MPI_Comm_split(MPI_COMM_CLUSTER, mycol, myrow, &Col_comm);

  // Allocate Degree array
  degree = (LOCINT *)Malloc(col_bl * sizeof(*degree));
  reach = (LOCINT *)Malloc(row_pp * sizeof(*reach));

  dist_lcc = (float *)Malloc(
      col_bl * sizeof(float)); // Local Cluster Coef... Store in scan mode
  if (heuristic == 1) {
    // if(myid==0) printf("task %d edges removed %d ...\n",gmyid,rem_ed);
    MPI_Allreduce(MPI_IN_PLACE, &rem_ed, 1, LOCINT_MPI, MPI_SUM,
                  MPI_COMM_CLUSTER);
    MPI_Allreduce(MPI_IN_PLACE, reach, row_pp, MPI_INT, MPI_SUM, Row_comm);
    if (myid == 0)
      printf("task %d Total edges removed %d\n", gmyid, rem_ed);
  }

  // get_deg(degree);
  // Calculate degree
  MPI_Allreduce(MPI_IN_PLACE, degree, col_bl, MPI_INT, MPI_SUM, Col_comm);

  if (analyze_degree == 1)
    analyze_deg(degree, col_bl);

#ifdef _FINE_TIMINGS
  // Allocate for statistical data
  mystats = (STATDATA *)Malloc(N * sizeof(STATDATA));
  memset(mystats, 0, N * sizeof(STATDATA));
#endif



#ifdef SERIAL
if (myid == 0) fprintf(stdout, "Computing LCC using %d process on %d core per process\n", ntask, 1);
TIMER_START(0);
lcc_func(col, row, dist_lcc);
TIMER_STOP(0);

#elif HAVE_SIMD
#pragma omp parallel
{
		nthreads = omp_get_num_threads();
}
if (myid == 0) fprintf(stdout, "Computing LCC using %d process on %d core per process\n", ntask, nthreads);
TIMER_START(0);
lcc_func_bin_simd(col, row, dist_lcc);
TIMER_STOP(0);
#endif


if (myid == 0) fprintf(stdout, "LCC done in %f secs on %d rank\n",TIMER_ELAPSED(0)/1.0E+6, myid);

  MPI_Barrier(MPI_COMM_WORLD);
  if (gmyid == 0) {
    fprintf(stdout, "System summary:\n Total(gntask) GPUs %d - Total(fd) ntask "
                    "%d - Total(fr)  cntask %d\n",
            gntask, ntask, cntask);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (outdebug != NULL)
    fclose(outdebug);
  fprintf(stdout,
          "WNODE Global-ID %d - Cluster-ID %d -  Local-ID %d ... closing\n",
          gmyid, color, myid);
  if (mycol == 0 && resname != NULL) {
    FILE *resout = fopen(resname, "w");

    LOCINT k;
    for (k = 0; k < row_pp; k++) {
      fprintf(resout, "%d\t%.2f\n", LOCI2GI(k), dist_lcc[k]);
    }
    fclose(resout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  freeMem(col);
  freeMem(row);
  freeMem(mystats);
  freeMem(gfile);
  freeMem(degree);
  freeMem(reach);
  freeMem(vRbuf);
  freeMem(vRnum);
  freeMem(hSbuf);
  freeMem(hSnum);
  freeMem(hRbuf);
  freeMem(hRnum);
  freeMem(hSFbuf);
  freeMem(hRFbuf);
  // 2-degree

  // ONEPREFIX

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(Row_comm);
  MPI_Barrier(Col_comm);
  MPI_Comm_free(&Row_comm);
  MPI_Comm_free(&Col_comm);
  MPI_Comm_free(&MPI_COMM_CLUSTER);

#ifdef HAVE_LIBLSB
  LSB_Finalize();
#endif

  MMPI_FINALIZE();
  exit(EXIT_SUCCESS);
}
