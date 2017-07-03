package org.apache.flink.examples.java.oldExamples;


import com.google.common.primitives.Ints;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.flink.api.common.functions.GPUMapFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.MutableObjectIterator;


import java.io.IOException;
import java.util.ArrayList;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

public class GPUAddition extends GPUMapFunction<Integer, Integer> {

	private final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/output/NumberAddition.ptx";

	private CUdeviceptr pInputs;
	private CUdeviceptr pOutputs;
	private CUdeviceptr pSize;
	private CUfunction addOne;

	private int sizeOfValues;

	@Override
	public Integer cpuMap(Integer value) {
		return value + 1;
	}

	@Override
	public Integer[] gpuMap(ArrayList<Integer> values) {

		if(values.size() == 0){
			return new Integer[0];
		}


		// Copy over input to device memory
		cuMemcpyHtoD(pInputs, Pointer.to(Ints.toArray(values)), sizeOfValues * Sizeof.INT);
		cuMemcpyHtoD(pSize, Pointer.to(new int[] {sizeOfValues}), Sizeof.INT);

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

		Integer[] result = new Integer[values.size()];

		for(int i = 0; i < outputs.length; i++){
			result[i] = outputs[i];
		}

		return result;
	}

	@Override
	public void releaseResources() {
		cuMemFree(pInputs);
		cuMemFree(pOutputs);
		cuMemFree(pSize);
	}

	@Override
	public void setDataProcessingTime(long time) {

	}

	@Override
	public void initialize(int size) throws IOException {

		// Enable Exceptions
		JCudaDriver.setExceptionsEnabled(true);

		// Initialize the driver and create a context for the first device.
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);

		// Spawn a thread for each value
		sizeOfValues = size;

		// Load the ptx file.
		CUmodule module = new CUmodule();
		cuModuleLoad(module, moduleLocation);

		// Obtain a function pointer to the kernel function.
		addOne = new CUfunction();
		cuModuleGetFunction(addOne, module, "add_one");


		// Create pointers
		pInputs = new CUdeviceptr();
		pOutputs = new CUdeviceptr();
		pSize = new CUdeviceptr();


		// Allocating arrays for GPU
		cuMemAlloc(pInputs, sizeOfValues * Sizeof.INT);
		cuMemAlloc(pOutputs, sizeOfValues * Sizeof.INT);
		cuMemAlloc(pSize, Sizeof.INT);
	}

}
