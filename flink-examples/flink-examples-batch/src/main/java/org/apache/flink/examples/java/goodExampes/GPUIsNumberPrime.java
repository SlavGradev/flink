package org.apache.flink.examples.java.goodExampes;

import com.google.common.primitives.Ints;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.flink.api.common.functions.GPUMapFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.MutableObjectIterator;

import java.util.ArrayList;

import static jcuda.driver.JCudaDriver.*;

public class GPUIsNumberPrime extends GPUMapFunction<Integer, Boolean> {

	private final int blockSize = 512;
	private final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/output/IsNumberPrime.ptx";

	private int size;

	private CUdeviceptr pInputs;
	private CUdeviceptr pOutputs;
	private CUdeviceptr pSize;
	private Pointer kernelParameters;
	private CUfunction cubed;

	private int[] values;
	private MutableObjectIterator<Integer> input;
	private Collector<Boolean> collector;


	@Override
	public Boolean cpuMap(Integer value) {

		if (value % 2 == 0) return false;
		int max = (value - 1) / 2;
		for (int i = 2; i < max; i++) {
			if (value % ((2 * i) + 1) == 0) return false;
		}
		return true;

	}

	@Override
	public Boolean[] gpuMap(ArrayList<Integer> values) {

		IsNumberPrimeExample.insrs[0] = System.nanoTime();

		if(values.size() == 0){
			return new Boolean[0];
		}

		// Copy over input to device memory
		IsNumberPrimeExample.insrs[1] = System.nanoTime();
		cuMemcpyHtoD(pInputs, Pointer.to(Ints.toArray(values)), values.size() * Sizeof.INT);
		cuMemcpyHtoD(pSize, Pointer.to(new int[] {values.size()}), Sizeof.INT);
		IsNumberPrimeExample.insrs[2] = System.nanoTime();

		// Call the kernel function.
		IsNumberPrimeExample.insrs[3] = System.nanoTime();
		cuLaunchKernel(cubed,
			(int)Math.ceil( (double) values.size() / blockSize),  1, 1,      // Grid dimension
			blockSize, 1, 1,      // Block dimension
			0, null,               // Shared memory size and stream
			kernelParameters, null // Kernel- and extra parameters
		);
		cuCtxSynchronize();
		IsNumberPrimeExample.insrs[4] = System.nanoTime();

		short[] outputs = new short[values.size()];

		// Copy from device memory to host memory
		IsNumberPrimeExample.insrs[5] = System.nanoTime();
		cuMemcpyDtoH(Pointer.to(outputs), pOutputs, values.size()  * Sizeof.SHORT);
		IsNumberPrimeExample.insrs[6] = System.nanoTime();

		Boolean[] result = new Boolean[values.size()];

		// 1 is true, 0 i false
		for(int i = 0; i < outputs.length; i++){
			result[i] = outputs[i] == 1;
		}

		IsNumberPrimeExample.insrs[7] = System.nanoTime();

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
		IsNumberPrimeExample.insrs[8] = time;
	}

	@Override
	public void initialize(int size) throws Exception {
		this.size = size;
		// Enable Exceptions
		JCudaDriver.setExceptionsEnabled(true);

		// Initialize the driver and create a context for the first device.
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		CUcontext context = new CUcontext();
		cuCtxCreate(context, 0, device);

		// Load the ptx file.
		CUmodule module = new CUmodule();
		cuModuleLoad(module, moduleLocation);

		// Obtain a function pointer to the kernel function.
		cubed = new CUfunction();
		cuModuleGetFunction(cubed, module, "is_prime");

		// Create pointers
		pInputs = new CUdeviceptr();
		pOutputs = new CUdeviceptr();
		pSize = new CUdeviceptr();

		// Start Data Transfer

		// Allocating arrays for GPU
		cuMemAlloc(pInputs, this.size * Sizeof.INT);
		cuMemAlloc(pOutputs, this.size * Sizeof.SHORT);
		cuMemAlloc(pSize, Sizeof.INT);

		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		kernelParameters = Pointer.to(
			Pointer.to(pInputs),
			Pointer.to(pOutputs),
			Pointer.to(pSize)
		);
	}
}
