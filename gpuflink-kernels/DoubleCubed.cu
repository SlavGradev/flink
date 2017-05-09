extern "C"
__global__ void double_cubed(double *in, double *out, int *n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < *n){ 
		out[i] = in[i] * in[i] * in[i];
	}
  	__syncthreads();
}
