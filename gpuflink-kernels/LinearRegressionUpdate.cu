extern "C"
__global__ void linear_regression_update(double *t0, double *t1, double *r0, double *r1, int *c){
	r0[blockIdx.x] = t0[blockIdx.x] / c[blockIdx.x];
	r1[blockIdx.x] = t1[blockIdx.x] / c[blockIdx.x];
  	__syncthreads();
}
