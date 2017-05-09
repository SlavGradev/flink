extern "C"
__global__ void linear_regression_sub_update(double *t0, 
					 double *t1, 
					 double *x, 
					 double *y,
					 int *r0, 
 					 int *r1){
	r0[blockIdx.x] = *t0 - 0.01 * ((*t0 + (*t1 * x[blockIdx.x])) - y[blockIdx.x]);
	r1[blockIdx.x] = *t1 - 0.01 * (((*t0 + (*t1 * x[blockIdx.x])) - y[blockIdx.x]) * x[blockIdx.x]);
  	__syncthreads();
}
