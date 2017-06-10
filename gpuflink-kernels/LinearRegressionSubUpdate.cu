extern "C"
__global__ void linear_regression_sub_update(double *t0, 
					 double *t1, 
					 double *x, 
					 double *y,
					 int *r0, 
 					 int *r1,
					 int *n){
 	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if(index < *n){
		r0[index] = *t0 - 0.01 * ((*t0 + (*t1 * x[index])) - y[index]);
		r1[index] = *t1 - 0.01 * (((*t0 + (*t1 * x[index])) - y[index]) * x[index]);
	}
  	__syncthreads();
}
