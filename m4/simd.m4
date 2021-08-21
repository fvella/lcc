AC_DEFUN([WITH_SIMD],
    [AC_ARG_WITH(simd, AC_HELP_STRING([--with-simd], [enables simd and openmp code.]))
    #echo "####### x${with_simd}"
    if test "x${with_simd}" == "xyes"; then
        AC_DEFINE(HAVE_SIMD, 1, enables simd and openmp code)
        AC_MSG_NOTICE([SIMD support enabled])
        CFLAGS="$CFLAGS -fopenmp"
    fi
    ]
)
