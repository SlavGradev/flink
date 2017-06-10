extern "C"
__global__ void linear_regression_update(double *t0, double *t1, double *r0, double *r1, int *c, int *n){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < *n){
		r0[index] = t0[index] / c[index];
		r1[index] = t1[index] / c[index];
	}
  	__syncthreads();
}
