extern "C"
__global__ void is_prime(int *in, short *out, int *n){

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int value = in[index];

	if(index < *n){

		if (value % 2 == 0){
			out[index] = 0;
			return;
		}

		int max = (value - 1) / 2;

		for (int i = 2; i < max; i++) {
			if (value % ((2 * i) + 1) == 0){
				out[index] = 0;
				return;
			}
		}

		out[index] = 1;
	}
  	__syncthreads();
}
