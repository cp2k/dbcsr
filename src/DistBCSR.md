# *DistBCSR*: C++ Wrapper-Class for 'dbcsr'
---
---


## MPI-Environment: `DBCSR_Environment`
---

The class `DBCSR_Environment` contains the essential information
for the MPI-setup for a specific bcsr-blocking.
It contains the following objects (all public):

+ `std::string env_id`: identifier
+ `int dbcsr_dims[2]`, `int dbcsr_periods[2]`, `int dbcsr_coords[2]`, `int dbcsr_reorder`:
  MPI grid-information
+ `void* dbcsr_dist`: DBCSR distribution handle
+ `MPI_Comm dbcsr_group`: MPI comm-group
+ `int nblk_row`: number of grid-rows
+ `int nblk_col`: number of grid-columns
+ `std::vector<int> dbcsr_row_dist(nblk_row)`: holds rank of MPI-node for *i*th row
+ `std::vector<int> dbcsr_col_dist(nblk_col)`: holds rank of MPI-node for *i*th column
+ `double dbcsr_default_thresh`: default sparse threshold
+ `int mpi_rank`: rank
+ `int mpi_size`: number of MPI-nodes

### Constructor
`DBCSR_Environment(int nblk_row, int nblk_col, const std::string& env_id)` generates an
environment for `nblk_row` row-blocks and `nblk_col` column-blocks. The ID
defaults to `dbcsr_env_default` and can be used to assert that operations with
several distributed matrices use the same distribution pattern.
Note that so far only a straightforward round-robin distribution is employed.


## Distributed BCSR-Matrix: `DistBCSR`
---
Each object of `DistBCSR` contains a pointer to a valid `DBCSR_Environment`-object which
allows a compact representation of matrix operations. 

### Constructor
+ `DistBCSR(DBCSR_Environment* dbcsr_env, const std::string& mname="default matrix name")`
+ `DistBCSR(DistBCSR& ref)`, `DistBCSR(const DistBCSR& ref)`
+ `DistBCSR(size_t nrow, size_t ncol, std::vector<int>& row_dims, std::vector<int>& col_dims, DBCSR_Environment* dbcsr_env,
            const std::string& mname="default matrix name")`
+ `DistBCSR(size_t ldim, std::vector<int>& dims, DBCSR_Environment* dbcsr_env, bool add_zero_diag=false, const std::string& mname="default matrix name")`

with:

+ `nrow`/`ncol`: number of rows/columns of dense matrix
+ `row_dims`/`col_dims`: block-dimensions of row/column
+ `dbcsr_env`: pointer to object of class `DBCSR_Environment`

### Destructor
`~DistBCSR()`: releases the dbcsr-matrix.

### Methods
+ `void load(double const* src, double cthr=-1.e0)`: distribute dense matrix `src`
+ `void load(std::vector<double>& src, double cthr=-1.e0)`: distribute dense matrix `src`
+ `std::vector<double> gather()`: return dense matrix
+ `void mult(char mA, char mB, const DistBCSR& A, const DistBCSR& B, double alpha=1.e0, double beta=0.e0, double cthr=-1.e0)`: matrix-matrix multiplication
+ `void copy(const DistBCSR& src)`: copy content of matrix `src`
+ `void add(const DistBCSR& A)`: add matrix `A`
+ `void sub(const DistBCSR& A)`: subtract matrix `A`
+ `void add(const DistBCSR& A, const DistBCSR& B)`: store sum of matrices `A` and `B`
+ `void sub(const DistBCSR& A, const DistBCSR& B)`: store difference of matrices `A` and `B`
+ `void axpy(const DistBCSR& A, const double fac)`: evaluate `this += fac * matrix[A]`
+ `void axpy(const DistBCSR& A, const DistBCSR& B, const double fac)`: store axpy-result
+ `double trace()`: return trace of matrix
+ `void set_diag(const double val)`: set all diagonal elements to `val`
+ `void set_diag(double const* dvals)`: set diagonal element `matrix(i,i)` to `dvals(i)`
+ `void set_diag(const std::vector<double>& dvals)`: set diagonal element `matrix(i,i)` to `dvals(i)`
+ `void set(const double val)`: set all elements to `val`
+ `void zero()`: zero matrix
+ `double dot(const DistBCSR& A)`: return dot-product
+ `void filter(double eps=-1.e0)`: re-compress sparse-matrix wrt to threshold `eps`
+ `void scale(const double fac)`: scale matrix with factor `fac`
+ `void load(const std::string& cfname)`: load matrix from disk
+ `void write(const std::string& cfname)`: store matrix to disk
+ `double maxabs()`: return max. absolute value
+ `void hadamard(const DistBCSR& rhs)`: evaluate Hadamard-product
+ `void hadamard_inv(const DistBCSR& rhs)`: element-wise division (if |rhs| > 0)
+ `void gershgorin_estimate(double& eps0, double& epsn)`: Evaluate gershgorin-estimate
+ `dm_dbcsr& get_dbcsr()`: returns handle to FORTRAN-object

### Overloaded Operators
+ `DistBCSR& operator=(const DistBCSR& rhs)`
+ `DistBCSR& operator=(DistBCSR&& rhs)`
+ `DistBCSR& operator+=(const DistBCSR& rhs)`
+ `DistBCSR& operator-=(const DistBCSR& rhs)`
+ `DistBCSR& operator*=(const double& fac)`
+ `DistBCSR operator*(const double& fac)`
+ `DistBCSR& operator/=(const double& fac)`
+ `DistBCSR operator+(const DistBCSR& rhs) const`
+ `DistBCSR operator-(const DistBCSR& rhs) const`
+ `DistBCSR operator*(const DistBCSR& rhs) const`

## Examples
---
In folder `examples/` are two files that use most of the class-functionalities:

+ `DistBCSR_example_1`: basic methods
+ `DistBCSR_example_2`: overloaded operators
