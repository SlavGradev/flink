extern "C"
__global__ void add_one(int *in, int *out, int *n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < *n){ 
		out[i] = in[i] + 1;
	}
  	__syncthreads();
}
