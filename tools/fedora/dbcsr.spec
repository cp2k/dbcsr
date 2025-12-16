# Currently does not build with opencl/libxsmm
%bcond_with opencl

# No openmpi on i668 with openmpi 5 in Fedora 40+
%if 0%{?fedora} >= 40
%ifarch %{ix86}
%bcond_with openmpi
%else
%bcond_without openmpi
%endif
%else
%bcond_without openmpi
%endif

Name:          dbcsr
Version:       0.0.0
Release:       %autorelease
Summary:       Distributed Block Compressed Sparse Row matrix library
License:       GPL-2.0-or-later
URL:           https://cp2k.github.io/dbcsr/develop/
Source0:       https://github.com/cp2k/dbcsr/releases/download/v%{version}/dbcsr-%{version}.tar.gz

BuildRequires: cmake
BuildRequires: gcc-c++
BuildRequires: gcc-gfortran
BuildRequires: make
BuildRequires: flexiblas-devel
%if %{with opencl}
BuildRequires: libxsmm-devel
%endif
BuildRequires: python3-fypp

%global _description %{expand:
DBCSR stands for Distributed Blocked Compressed Sparse Row.

DBCSR is a library designed to efficiently perform sparse matrix-matrix
multiplication, among other operations.  It is MPI and OpenMP parallel and
can exploit Nvidia and AMD GPUs via CUDA and HIP.}


%description
%{_description}

This package contains the non-MPI single process and multi-threaded versions.

%package devel
Summary: Development files for %{name}
Requires: %{name}%{?_isa} = %{version}-%{release}
Requires: gcc-gfortran%{_isa}

%description devel
The %{name}-devel package contains libraries and header files for
developing applications that use %{name}.

%global mpi_list mpich

%if %{with openmpi}
%global mpi_list %{mpi_list} openmpi
%package openmpi
Summary: DBCSR - openmpi version
BuildRequires:  openmpi-devel

%description openmpi
%{_description}

This package contains the parallel single- and multi-threaded versions
using OpenMPI.

%package openmpi-devel
Summary: Development files for %{name}-openmpi
Requires: %{name}-openmpi%{?_isa} = %{version}-%{release}

%description openmpi-devel
The %{name}-openmpi-devel package contains libraries and header files for
developing applications that use %{name}-openmpi.
%endif

%package mpich
Summary: DBCSR - mpich version
BuildRequires:  mpich-devel

%description mpich
%{_description}

This package contains the parallel single- and multi-threaded versions
using mpich.

%package mpich-devel
Summary: Development files for %{name}-mpich
Requires: %{name}-mpich%{?_isa} = %{version}-%{release}

%description mpich-devel
The %{name}-mpich-devel package contains libraries and header files for
developing applications that use %{name}-mpich.


%prep
%autosetup -p1
# Use system fypp, other tools not needed
rm -r tools


# $mpi will be evaluated in the loops below
%global _vpath_builddir %{_vendor}-%{_target_os}-build-${mpi:-serial}

%build
export CFLAGS="%{optflags} -fPIC"
export CXXFLAGS="%{optflags} -fPIC"
export FFLAGS="%{optflags} -fPIC"
%cmake \
  -DCMAKE_INSTALL_Fortran_MODULES=%{_fmoddir} \
  -DUSE_MPI=OFF \
  %{?with_opencl:-DUSE_ACCEL=opencl -DUSE_SMM=libxsmm}
%cmake_build
for mpi in %{mpi_list}
do
  module load mpi/$mpi-%{_arch}
  %cmake \
    -DCMAKE_INSTALL_Fortran_MODULES=$MPI_FORTRAN_MOD_DIR \
    %{?with_opencl:-DUSE_ACCEL=opencl -DUSE_SMM=libxsmm} \
    -DCMAKE_INSTALL_PREFIX:PATH=$MPI_HOME \
    -DCMAKE_INSTALL_LIBDIR:PATH=$MPI_LIB \
    -DUSE_MPI_F08=ON \
    -DTEST_MPI_RANKS=2
  %cmake_build
  module purge
done


%install
%cmake_install
for mpi in %{mpi_list}
do
  module load mpi/$mpi-%{_arch}
  %cmake_install
  module purge
done


%check
%ctest
for mpi in %{mpi_list}
do
  module load mpi/$mpi-%{_arch}
  fail=
  # mpich tests fail on s390x - reported https://github.com/cp2k/dbcsr/issues/703
  [ $mpi = mpich -a %{_arch} = s390x ] && fail=no
  %ctest || test $fail
  module purge
done


%files
%license LICENSE
%doc README.md
%{_libdir}/libdbcsr.so.*

%files devel
%{_fmoddir}/dbcsr_api.mod
%{_fmoddir}/dbcsr_tensor_api.mod
%{_libdir}/cmake/dbcsr/
%{_libdir}/libdbcsr.so

%if %{with openmpi}
%files openmpi
%license LICENSE
%doc README.md
%{_libdir}/openmpi/lib/libdbcsr.so.*
%{_libdir}/openmpi/lib/libdbcsr_c.so.*

%files openmpi-devel
%{_libdir}/openmpi/include/dbcsr.h
%{_libdir}/openmpi/include/dbcsr_tensor.h
%{_fmoddir}/openmpi/dbcsr_api.mod
%{_fmoddir}/openmpi/dbcsr_tensor_api.mod
%{_libdir}/openmpi/lib/cmake/dbcsr/
%{_libdir}/openmpi/lib/libdbcsr.so
%{_libdir}/openmpi/lib/libdbcsr_c.so
%endif

%files mpich
%license LICENSE
%doc README.md
%{_libdir}/mpich/lib/libdbcsr.so.*
%{_libdir}/mpich/lib/libdbcsr_c.so.*

%files mpich-devel
%{_libdir}/mpich/include/dbcsr.h
%{_libdir}/mpich/include/dbcsr_tensor.h
%{_fmoddir}/mpich/dbcsr_api.mod
%{_fmoddir}/mpich/dbcsr_tensor_api.mod
%{_libdir}/mpich/lib/cmake/dbcsr/
%{_libdir}/mpich/lib/libdbcsr.so
%{_libdir}/mpich/lib/libdbcsr_c.so

%changelog
%autochangelog
