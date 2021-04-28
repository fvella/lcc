#define OMP_SCHEDULING "omp parallel for schedule(static)"

// #define RANDOM_REORDER 1

#ifdef DEGBSD_REORDER
    #define REORDER "degbsd"
#elif defined(RANDOM_REORDER)
    #define REORDER "random"
#else
    #define REORDER "in-order"
#endif
