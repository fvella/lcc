AC_DEFUN([WITH_SIMD],
    [AC_MSG_NOTICE([*** checking for OPENMP/SIMD support ***])
        [AC_ARG_ENABLE(simd, AC_HELP_STRING([--with-simd], [compile with OpenMP support]))
        printf "${with_simd}"
        AC_MSG_NOTICE(using -fopenmp CFLAG)
        AC_DEFINE(HAVE_SIMD, 1, enables the specific code)
        have_simd=1
    ]

