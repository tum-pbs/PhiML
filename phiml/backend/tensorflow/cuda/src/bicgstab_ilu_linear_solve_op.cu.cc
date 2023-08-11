
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cusolverSp.h>
#include <cusparse.h>
//#include <cublas.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cuda_runtime_api.h>


static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;

  exit(1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
inline void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %p %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

#define cdpErrchk_sparse(ans) { cdpAssert_sparse((ans), __FILE__, __LINE__); }
inline void cdpAssert_sparse(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSPARSE_STATUS_SUCCESS)
   {
      printf("GPU kernel assert: %i %p %d\n", code, file, line);
      if (abort) assert(0);
   }
}

#define cdpErrchk_blas(ans) { cdpAssert_blas((ans), __FILE__, __LINE__); }
inline void cdpAssert_blas(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS)
   {
      printf("GPU kernel assert: %i %p %d\n", code, file, line);
      if (abort) assert(0);
   }
}


#define cdpErrchk_solver(ans) { cdpAssert_solver((ans), __FILE__, __LINE__); }
inline void cdpAssert_solver(cusolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSOLVER_STATUS_SUCCESS)
   {
      printf("GPU kernel assert: %i %p %d\n", code, file, line);
      if (abort) assert(0);
   }
}


__global__ void printarray(float* a, int dim){
  for (size_t i = 0; i < dim; i++) {
    printf(" %f ", a[i]);
  }
  printf("\n");
}

__global__ void printarray_int(int* a, int dim){
  for (size_t i = 0; i < dim; i++) {
    printf(" %d ", a[i]);
  }
  printf("\n");
}

__global__ void print_section(float* cval, int* rptr, int dim){
  for (size_t i = rptr[dim-1];i < rptr[dim]; i++) {
    printf(" %f ", cval[i]);
  }
  printf("\n");
}


__global__ void print_section_int(int* cval, int* rptr, int dim){
  for (size_t i = rptr[dim-1];i < rptr[dim]; i++) {
    printf(" %d ", cval[i]);
  }
  printf("\n");
}

#define CUSPARSE_ALG CUSPARSE_ALG_NAIVE
#define DTYPE CUDA_R_32F
typedef float dtype;

// Assumes symmetric sparsity pattern (might also be filled with zeros)
__global__ void transpose_csr_into_cpy(dtype* csr_values, int* csr_row_ptr, int* csr_col_ind, dtype* csr_values_dest, const int matrix_shape){

  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < matrix_shape; row += blockDim.x * gridDim.x)
  {
      int j;
      for (int i = csr_row_ptr[row]; i < csr_row_ptr[row+1]; i++){
        j = csr_row_ptr[csr_col_ind[i]];
        while(csr_col_ind[j]!=row) j++;
        csr_values_dest[j] = csr_values[i];
      }
  }
}


__host__ void BicgstabIluLinearSolveLauncher(dtype * csr_valuesA, int* csr_row_ptr, int* csr_col_ind,
    const dtype* rhs, const int nnz_a, const int batch_size, const int matrix_shape,const dtype *x_old,
  dtype tol, int max_it,
  dtype *s, dtype *s_hat,  dtype *p, dtype *p_hat, dtype *r, dtype *rh, dtype *v, dtype *t, dtype *z, dtype *csr_values, dtype * x,
  const bool transpose_operation)
{
  cusparseOperation_t transpose_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  /*if (transpose_operation == true){
    transpose_op = CUSPARSE_OPERATION_TRANSPOSE;
  }
  else {
    transpose_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  }**/
  // TO BE REOMVED -- PRINT ALL THEM POINTERS
  /*printf("csrA %p\n", csr_valuesA);
  printf("csrrp %p\n", csr_row_ptr);
  printf("csrid %p\n", csr_col_ind);
  printf("csrilu %p\n", csr_values);
  printf("rhs %p\n", rhs);
  printf("x %p\n", x);
  printf("s %p\n", s);
  printf("s_hat %p\n", s_hat);
  printf("p %p\n", p);
  printf("p_hat %p\n", p_hat);
  printf("r %p\n", r);
  printf("rh %p\n", rh);
  printf("v %p\n", v);
  printf("t %p\n", t);
  printf("z %p\n", z);*/

  cublasHandle_t b_handle = NULL;
  cdpErrchk_blas(cublasCreate(&b_handle));

  // Copy csr_values in new array since iLU overwrites array
  dtype* csr_values_transposed;
  dtype* temp;
  if (transpose_operation){
    int blockSize;
    int minGridSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(  &minGridSize, &blockSize, transpose_csr_into_cpy, 0, 0 );
    gridSize = (matrix_shape + blockSize -1) / blockSize;
    transpose_csr_into_cpy<<<gridSize,blockSize>>>(csr_valuesA, csr_row_ptr, csr_col_ind, csr_values, matrix_shape);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    cdpErrchk(cudaMalloc((void**) &csr_values_transposed, nnz_a*sizeof(dtype)));
    cdpErrchk_blas(cublasScopy(b_handle, nnz_a, csr_values, 1, csr_values_transposed, 1));
    temp = csr_valuesA;
    csr_valuesA = csr_values_transposed;
  }
  else{
      cdpErrchk_blas(cublasScopy(b_handle, nnz_a, csr_valuesA, 1, csr_values, 1));
  }
  cdpErrchk_blas(cublasScopy(b_handle, matrix_shape, x_old, 1, x, 1));
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());

  cusparseHandle_t handle = NULL;
  cdpErrchk_sparse(cusparseCreate(&handle));
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

  cusparseMatDescr_t descrA = NULL;
  cusparseMatDescr_t descrL = NULL;
  cusparseMatDescr_t descrU = NULL;

  csrilu02Info_t infoA = NULL;
  csrsv2Info_t  infoL  = NULL;
  csrsv2Info_t  infoU  = NULL;

  int pBufferSizeA;
  int pBufferSizeL;
  int pBufferSizeU;
  int pBufferSize;
  void *pBuffer = NULL;

  int structural_zero;
  int numerical_zero;
  const cusparseSolvePolicy_t policyA = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policyL = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policyU = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t transL  = transpose_op;
  const cusparseOperation_t transU  = transpose_op;

  cdpErrchk_sparse(cusparseCreateMatDescr(&descrA));
  cdpErrchk_sparse(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  cdpErrchk_sparse(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

  cdpErrchk_sparse(cusparseCreateMatDescr(&descrL));
  cdpErrchk_sparse(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL));
  cdpErrchk_sparse(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
  cdpErrchk_sparse(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
  cdpErrchk_sparse(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT));

  cdpErrchk_sparse(cusparseCreateMatDescr(&descrU));
  cdpErrchk_sparse(cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL));
  cdpErrchk_sparse(cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO));
  cdpErrchk_sparse(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER));
  cdpErrchk_sparse(cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));

  cdpErrchk_sparse(cusparseCreateCsrilu02Info(&infoA));
  cdpErrchk_sparse(cusparseCreateCsrsv2Info(&infoL));
  cdpErrchk_sparse(cusparseCreateCsrsv2Info(&infoU));

  cusparseScsrilu02_bufferSize(handle, matrix_shape, nnz_a,
    descrA, csr_values, csr_row_ptr, csr_col_ind, infoA, &pBufferSizeA);
  cusparseScsrsv2_bufferSize(handle, transL, matrix_shape, nnz_a,
    descrL, csr_values, csr_row_ptr, csr_col_ind, infoL, &pBufferSizeL);
  cusparseScsrsv2_bufferSize(handle, transU, matrix_shape, nnz_a,
    descrU, csr_values, csr_row_ptr, csr_col_ind, infoU, &pBufferSizeU);

  pBufferSize = max(pBufferSizeA,max(pBufferSizeL,pBufferSizeU));
  cdpErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));

  //printf("LU buffer allocated\n");

  cdpErrchk_sparse(cusparseScsrilu02_analysis(handle, matrix_shape, nnz_a, descrA,
    csr_values, csr_row_ptr, csr_col_ind, infoA, policyA, pBuffer));
  cusparse_status = cusparseXcsrilu02_zeroPivot(handle, infoA, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status){
    printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    print_section<<<1,1>>>(csr_values, csr_row_ptr, structural_zero+1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    print_section_int<<<1,1>>>(csr_col_ind, csr_row_ptr,structural_zero+1);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

      //printf("A(%d,%d) is missing\n", structural_zero, structural_zero);

  }

  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  cdpErrchk_sparse(cusparseScsrilu02(handle, matrix_shape, nnz_a, descrA,
    csr_values, csr_row_ptr, csr_col_ind, infoA, policyA, pBuffer));
  cusparse_status = cusparseXcsrilu02_zeroPivot(handle, infoA, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == cusparse_status){
     printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
  }

//printf("LU factorisation created\n");
  /* NON- FUNCTIONING LU SOLVE - -Zero fill in LU too inaccurate

  cusparseScsrsv2_analysis(handle, transL, matrix_shape, nnz_a, descrL,
    csr_values, csr_row_ptr, csr_col_ind,
    infoL, policyL, pBuffer);

  cusparseScsrsv2_analysis(handle, transU, matrix_shape, nnz_a, descrU,
    csr_values, csr_row_ptr, csr_col_ind,
    infoU, policyU, pBuffer);

  const float alpha = 1.;
  printf("firstsolve\n");
  // step 6: solve L*z = rhs
  cusparse_status = cusparseScsrsv2_solve(handle, transL, matrix_shape, nnz_a, &alpha, descrL,
     csr_values, csr_row_ptr, csr_col_ind, infoL,
     rhs, z, policyL, pBuffer);
  printf("firstsolve\n");

  printarray<<<1,1>>>(z,matrix_shape);
  printf("seconsolve\n");
  // step 7: solve U*x = z
cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transU, matrix_shape, nnz_a, &alpha, descrU,
     csr_values, csr_row_ptr, csr_col_ind, infoU,
     z, x, policyU, pBuffer));

  printarray<<<1,1>>>(x,matrix_shape);
*/



cusparseScsrsv2_analysis(handle, transL, matrix_shape, nnz_a, descrL,
  csr_values, csr_row_ptr, csr_col_ind,
  infoL, policyL, pBuffer);

cusparseScsrsv2_analysis(handle, transU, matrix_shape, nnz_a, descrU,
  csr_values, csr_row_ptr, csr_col_ind,
  infoU, policyU, pBuffer);

void *bicgBuffer = NULL;
size_t bicgBufferSize;

dtype help = 1.;
dtype help2 = 0.;

dtype alpha = 1.;
dtype rho = 1.;
dtype rhop = 1.;
dtype omega = 1.;
dtype beta;
dtype nrm_r;
/*
dtype *s;
dtype *s_hat;
dtype *p;
dtype *p_hat;
dtype *r;
dtype *rh;
dtype *v;
dtype *t;
dtype *z;
cdpErrchk(cudaMalloc((void**) &s,     matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &s_hat, matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &p,     matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &p_hat, matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &r,     matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &rh,    matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &v,     matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &t,     matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &z,     matrix_shape*sizeof(dtype)));*/

/*
dtype *helpvec;
dtype *helpvec2;
cdpErrchk(cudaMalloc((void**) &helpvec,     matrix_shape*sizeof(dtype)));
cdpErrchk(cudaMalloc((void**) &helpvec2,     matrix_shape*sizeof(dtype)));
*/

/*// test L solve
printf("X vector:\n");
printarray<<<1,1>>>(x, matrix_shape);
CUDA_CHECK_RETURN(cudaDeviceSynchronize());

cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
 csr_values, csr_row_ptr, csr_col_ind, infoL,
 x, z, policyL, pBuffer));

 printarray<<<1,1>>>(z,matrix_shape);
 CUDA_CHECK_RETURN(cudaDeviceSynchronize());*/


// COMPUTE RESIDUAL r = b - A * X_0
  // r = A * x_0 + 0.*r
cdpErrchk_sparse(cusparseCsrmvEx_bufferSize( handle,
                            CUSPARSE_ALG,
                            transpose_op,
                            matrix_shape, matrix_shape,
                            nnz_a,
                            &help, DTYPE,
                            descrA,
                            csr_valuesA, DTYPE,
                            csr_row_ptr,
                            csr_col_ind,
                            x, DTYPE,
                            &help2,  DTYPE,
                            r,   DTYPE,
                            DTYPE,
                            &bicgBufferSize));

cdpErrchk(cudaMalloc((void**)&bicgBuffer, bicgBufferSize));

cdpErrchk_sparse(cusparseCsrmvEx(  handle,
                  CUSPARSE_ALG,
                  transpose_op,
                  matrix_shape, matrix_shape,
                  nnz_a,
                  &help, DTYPE,
                  descrA,
                  csr_valuesA, DTYPE,
                  csr_row_ptr,
                  csr_col_ind,
                  x,   DTYPE,
                  &help2,  DTYPE,
                  r,   DTYPE,
                  DTYPE,
                  bicgBuffer));
  // r = b - r
help2 = -1.;
cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &help2, r, 1));
help = 1.;
cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &help, rhs, 1, r, 1));

// initial guess lucky Check
cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
//printf("norm %f\n", nrm_r);
if (nrm_r < tol){
  goto endofloop;
}

// copy p = r  &  r_hat = r
cdpErrchk_blas(cublasScopy(b_handle, matrix_shape, r, 1, p, 1));
cdpErrchk_blas(cublasScopy(b_handle, matrix_shape, r, 1, rh, 1));

// initialise v & p with zeros
help2 = 0.;
cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &help2, v, 1));
cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &help2, p, 1));


// MAIN BICGSTAB LOOP
for (int i = 0; i < max_it; i++) {
  /*printf("iteration %d\n",i );
  printf("x\n");
  printarray<<<1,1>>>(x,matrix_shape);
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  printf("r\n" );
  printarray<<<1,1>>>(r,matrix_shape);
  printf("rh\n" );
  printarray<<<1,1>>>(rh,matrix_shape);
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());*/

  // rho = <r_hat, r>
  rhop = rho;
  cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, r, 1, rh, 1, &rho));
  //12: beta = (rho_{i} / rho_{i-1}) ( alpha / \mega )
  beta= (rho/rhop)*(alpha/omega);
  omega = -omega;
  help = 1.0;
  //13: p = r + beta * (p - omega * v)
  cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &omega, v, 1, p, 1));
  cdpErrchk_blas(cublasSscal(b_handle, matrix_shape, &beta, p, 1));
  cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &help,  r, 1, p, 1));
  omega = - omega;

    // solve L*z = rhs
  cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
     csr_values, csr_row_ptr, csr_col_ind, infoL,
     p, z, policyL, pBuffer));
  // solve U*x = z
  cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transU, matrix_shape, nnz_a, &help, descrU,
     csr_values, csr_row_ptr, csr_col_ind, infoU,
     z, p_hat, policyU, pBuffer));
     /*printf("phat\n");
     printarray<<<1,1>>>(p_hat,matrix_shape);*/
/*
     cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transU, matrix_shape, nnz_a, &help, descrU,
        csr_values, csr_row_ptr, csr_col_ind, infoU,
        p, p_hat, policyU, pBuffer));
*/

  // v = A * p_hat
  help = 1.;
  help2 = 0.;
  cdpErrchk_sparse(cusparseCsrmvEx(  handle,
                    CUSPARSE_ALG,
                    transpose_op,
                    matrix_shape, matrix_shape,
                    nnz_a,
                    &help, DTYPE,
                    descrA,
                    csr_valuesA, DTYPE,
                    csr_row_ptr,
                    csr_col_ind,
                    p_hat,   DTYPE,
                    &help2,  DTYPE,
                    v,   DTYPE,
                    DTYPE,
                    bicgBuffer));

  // alpha = rho_i / <r_hat,v>
  cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, rh, 1, v, 1, &alpha));
  alpha = rho/alpha;

  // x = x + alpha * p_hat
  cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &alpha, p_hat, 1, x, 1));

  // s = r - alpha * v    ::: S = R :::
  alpha = - alpha;
  cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &alpha, v, 1, r, 1));
  alpha = - alpha;
  // convergence Check
  cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
  //printf("norm %f\n", nrm_r);
  if (nrm_r < tol){
    break;
  }

  //M \hat{s} = r (sparse lower and upper triangular solves)
  help = 1.;

  // solve L*z = r
  cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
    csr_values, csr_row_ptr, csr_col_ind, infoL,
    r, z, policyL, pBuffer));
  // solve U*s_hat = z
  cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transU, matrix_shape, nnz_a, &help, descrU,
    csr_values, csr_row_ptr, csr_col_ind, infoU,
    z, s_hat, policyU, pBuffer));
    /*
    cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transU, matrix_shape, nnz_a, &help, descrU,
      csr_values, csr_row_ptr, csr_col_ind, infoU,
      r, s_hat, policyU, pBuffer));
*/

  // t = A * s_hat
  help2 = 0.;
  cdpErrchk_sparse(cusparseCsrmvEx(  handle,
                    CUSPARSE_ALG,
                    transpose_op,
                    matrix_shape, matrix_shape,
                    nnz_a,
                    &help, DTYPE,
                    descrA,
                    csr_valuesA, DTYPE,
                    csr_row_ptr,
                    csr_col_ind,
                    s_hat,   DTYPE,
                    &help2,  DTYPE,
                    t,   DTYPE,
                    DTYPE,
                    bicgBuffer));

  // omega = <t,s> / <t, t>
  /*
  cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
     csr_values, csr_row_ptr, csr_col_ind, infoL,
     t, helpvec, policyL, pBuffer));
  cdpErrchk_sparse(cusparseScsrsv2_solve(handle, transL, matrix_shape, nnz_a, &help, descrL,
    csr_values, csr_row_ptr, csr_col_ind, infoL,
    r, helpvec2, policyL, pBuffer));
  cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, helpvec, 1, helpvec2, 1, &help));
  cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, helpvec, 1, helpvec, 1, &help2));*/

  cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, t, 1, r, 1, &help));
  cdpErrchk_blas(cublasSdot(b_handle, matrix_shape, t, 1, t, 1, &help2));
  omega = help/help2;

  // x = x + omega * s_hat
  cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &omega, s_hat, 1, x, 1));

  // r = s - omega * t
  omega = -omega;
  cdpErrchk_blas(cublasSaxpy(b_handle, matrix_shape, &omega, t, 1, r, 1));
  omega = - omega;

  // convergence check
  cdpErrchk_blas(cublasSnrm2(b_handle, matrix_shape, r , 1, &nrm_r));
  //printf("norm %f\n", nrm_r);
  if (nrm_r < tol){
    break;
  }


}
endofloop:

  if (transpose_operation){
    csr_valuesA = temp;
    cudaFree(csr_values_transposed);
  }
  // _____________ NON FINAL: RETURN  LU FACTORISED ARRAY ___________________
  //  cdpErrchk_blas(cublasScopy(b_handle,nnz_a, csr_values, 1, csr_valuesA,1));
  //_______________________________________________________________________
  // step 6: free resources
  cudaFree(pBuffer);
  cudaFree(bicgBuffer);
  /*cudaFree(s);
  cudaFree(s_hat);
  cudaFree(p);
  cudaFree(p_hat);
  cudaFree(r);
  cudaFree(rh);
  cudaFree(v);
  cudaFree(t);
  cudaFree(z);*/
  //cudaFree(csr_values);
  cusparseDestroyMatDescr(descrA);
  cusparseDestroyMatDescr(descrL);
  cusparseDestroyMatDescr(descrU);
  cusparseDestroyCsrilu02Info(infoA);
  cusparseDestroyCsrsv2Info(infoL);
  cusparseDestroyCsrsv2Info(infoU);
  cusparseDestroy(handle);
  cublasDestroy(b_handle);
  //cudaFree(csr_values);


}
