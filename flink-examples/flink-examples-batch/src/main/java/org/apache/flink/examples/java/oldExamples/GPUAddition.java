package org.apache.flink.examples.java;


import com.google.common.primitives.Ints;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.flink.api.common.functions.GPUMapFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.MutableObjectIterator;


import java.util.ArrayList;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

public class GPUAddition extends GPUMapFunction<Integer, Integer> {

	private final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/NumberAddition.ptx";

	@Override
	public Integer cpuMap(Integer value) {
		return value + 1;
	}

	@Override
	public Integer[] gpuMap() {
		ArrayList<Integer> values = new ArrayList<>();
		if(values.size() == 0){
			return new Integer[0];
		}

		// Get the underlying array
		int[] inputs = Ints.toArray(values);

		// Enable Exceptions
		JCudaDriver.setExceptionsEnabled(true);

		// Initialize the driver and create a context for the first device.
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);

		// Spawn a thread for each value
		int sizeOfValues = values.size();

		// Load the ptx file.
		CUmodule module = new CUmodule();
		cuModuleLoad(module, moduleLocation);

		// Obtain a function pointer to the kernel function.
		CUfunction addOne = new CUfunction();
		cuModuleGetFunction(addOne, module, "add_one");


		// Create pointers
		CUdeviceptr pInputs = new CUdeviceptr();
		CUdeviceptr pOutputs = new CUdeviceptr();
		CUdeviceptr pSize = new CUdeviceptr();



		// Allocating arrays for GPU
		cuMemAlloc(pInputs, sizeOfValues * Sizeof.INT);
		cuMemAlloc(pOutputs, sizeOfValues * Sizeof.INT);
		cuMemAlloc(pSize, Sizeof.INT);


		// Copy over input to device memory
		cuMemcpyHtoD(pInputs, Pointer.to(inputs), sizeOfValues * Sizeof.INT);
		cuMemcpyHtoD(pInputs, Pointer.to(new int[] {sizeOfValues}), Sizeof.INT);

		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		Pointer kernelParameters = Pointer.to(
			Pointer.to(pInputs),
			Pointer.to(pOutputs),
			Pointer.to(pSize)
		);

		int blockSizeX = 256;
		int gridSizeX = (int)Math.ceil((double)values.size() / blockSizeX);

		// Call the kernel function.
		cuLaunchKernel(addOne,
			gridSizeX,  1, 1,      // Grid dimension
			blockSizeX, 1, 1,      // Block dimension
			0, null,               // Shared memory size and stream
			kernelParameters, null // Kernel- and extra parameters
		);

		cuCtxSynchronize();
		int[] outputs = new int[values.size()];

		// Copy from device memory to host memory
		cuMemcpyDtoH(Pointer.to(outputs), pOutputs, sizeOfValues * Sizeof.INT);
		cuCtxSynchronize();

		Integer[] result = new Integer[outputs.length];

		for(int i = 0; i < outputs.length; i++){
			result[i] = Integer.valueOf(outputs[i]);
		}

		cuMemFree(pInputs);
		cuMemFree(pOutputs);


		return result;
	}

	@Override
	public void releaseResources() {

	}

	@Override
	public void setDataProcessingTime(long time) {

	}

	@Override
	public void initialize(MutableObjectIterator<Integer> input, Collector<Integer> outputCollector) {

	}

}
