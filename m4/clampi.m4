AC_DEFUN([WITH_CLAMPI],
    [AC_ARG_WITH(clampi, AC_HELP_STRING([--with-clampi], [compile with CLaMPI]))
    clampi_found=no
    AC_SUBST([CLAMPI_CFLAGS], [])
    AC_SUBST([CLAMPI_LDFLAGS], [])

    if test x"${with_clampi}" == xyes; then
        AC_CHECK_HEADER(clampi.h, clampi_found=yes, [AC_MSG_ERROR([CLaMPI support selected but headers not available!])])
    elif test x"${with_clampi}" == xno; then
        clampi_found=no
    elif test x"${with_clampi}" != x; then
        CPPFLAGS="$CPPFLAGS -I${with_clampi}/include/"
        AC_CHECK_HEADER(clampi.h, [clampi_path=${with_clampi}; clampi_found=yes], [AC_MSG_ERROR([Can't find the CLaMPI header files in ${with_clampi}])])
    fi

    if test x"${clampi_found}" == xyes; then
        AC_DEFINE(HAVE_CLAMPI, 1, enables CLaMPI)
        AC_MSG_NOTICE([CLaMPI support enabled])
        AC_SUBST([CLAMPI_CFLAGS], [])
        AC_SUBST([CLAMPI_LDFLAGS], [])
        if test x${clampi_path} != x; then
            CXXFLAGS="${CXXFLAGS} -I${with_clampi}/include/"
            LDFLAGS="${LDFLAGS} -L${clampi_path}/lib/"
            AC_SUBST([CLAMPI_CFLAGS], [-I${clampi_path}/include/])
            AC_SUBST([CLAMPI_LDFLAGS], [-L${clampi_path}/lib/])
        fi
        LIBS="${LIBS} -lclampi"
    fi
    ]
)

