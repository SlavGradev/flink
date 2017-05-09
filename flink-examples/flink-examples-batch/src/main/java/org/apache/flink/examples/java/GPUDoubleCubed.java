package org.apache.flink.examples.java;


import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.flink.api.common.functions.GPUSupportingMapFunction;

import java.util.ArrayList;

import static jcuda.driver.JCudaDriver.*;

public class GPUDoubleCubed extends GPUSupportingMapFunction<Double, Double>{

	private final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/DoubleCubed.ptx";

	@Override
	public Double cpuMap(Double value) {
		return value * value * value;
	}

	@Override
	public Double[] gpuMap(ArrayList<Double> values) {
		if(values.size() == 0){
			return new Double[0];
		}

		// Get the underlying array
		double[] inputs = Doubles.toArray(values);

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
		CUfunction cubed = new CUfunction();
		cuModuleGetFunction(cubed, module, "double_cubed");

		// Create pointers
		CUdeviceptr pInputs = new CUdeviceptr();
		CUdeviceptr pOutputs = new CUdeviceptr();
		CUdeviceptr pSize = new CUdeviceptr();

		// Allocating arrays for GPU
		cuMemAlloc(pInputs, sizeOfValues * Sizeof.DOUBLE);
		cuMemAlloc(pOutputs, sizeOfValues * Sizeof.DOUBLE);
		cuMemAlloc(pSize, Sizeof.INT);


		// Copy over input to device memory
		cuMemcpyHtoD(pInputs, Pointer.to(inputs), sizeOfValues * Sizeof.DOUBLE);
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
		cuLaunchKernel(cubed,
			gridSizeX,  1, 1,      // Grid dimension
			blockSizeX, 1, 1,      // Block dimension
			0, null,               // Shared memory size and stream
			kernelParameters, null // Kernel- and extra parameters
		);

		cuCtxSynchronize();
		double[] outputs = new double[values.size()];

		// Copy from device memory to host memory
		cuMemcpyDtoH(Pointer.to(outputs), pOutputs, sizeOfValues * Sizeof.DOUBLE);
		cuCtxSynchronize();

		Double[] result = new Double[outputs.length];

		for(int i = 0; i < outputs.length; i++){
			result[i] = Double.valueOf(outputs[i]);
		}

		cuMemFree(pInputs);
		cuMemFree(pOutputs);

		return result;
	}

}
