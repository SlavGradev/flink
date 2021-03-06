package org.apache.flink.examples.java.goodExampes;


import com.google.common.primitives.Doubles;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.flink.api.common.functions.GPUMapFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.MutableObjectIterator;

import java.util.ArrayList;

import static jcuda.driver.JCudaDriver.*;

public class GPUDoubleCubed extends GPUMapFunction<Double, Double> {

	private int size;
	MutableObjectIterator<Double> input;
	Collector<Double> collector;
	double[] values;

	private CUdeviceptr pInputs;
	private CUdeviceptr pOutputs;
	private CUdeviceptr pSize;

	private Pointer kernelParameters;
	private CUfunction cubed;

	private final int blockSize = 512;

	private final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/output/DoubleCubed.ptx";

	@Override
	public Double cpuMap(Double value) {
		return value * value * value;
	}

	@Override
	public void initialize(int size) throws Exception {

	}

	@Override
	public Double[] gpuMap(ArrayList<Double> values) {


		DoubleCubedExample.insrs[0] = System.nanoTime();

		if(values.size() == 0){
			return new Double[0];
		}

		// Copy over input to device memory
		DoubleCubedExample.insrs[1] = System.nanoTime();
		cuMemcpyHtoD(pInputs, Pointer.to(Doubles.toArray(values)), values.size() * Sizeof.DOUBLE);
		cuMemcpyHtoD(pSize, Pointer.to(new int[] {values.size()}), Sizeof.INT);
		DoubleCubedExample.insrs[2] = System.nanoTime();

		// Call the kernel function.
		DoubleCubedExample.insrs[3] = System.nanoTime();
		cuLaunchKernel(cubed,
			(int)Math.ceil( (double) values.size() / blockSize),  1, 1,      // Grid dimension
			blockSize, 1, 1,      // Block dimension
			0, null,               // Shared memory size and stream
			kernelParameters, null // Kernel- and extra parameters
		);
		cuCtxSynchronize();
		DoubleCubedExample.insrs[4] = System.nanoTime();

		double[] outputs = new double[values.size()];

		// Copy from device memory to host memory
		DoubleCubedExample.insrs[5] = System.nanoTime();
		cuMemcpyDtoH(Pointer.to(outputs), pOutputs, values.size() * Sizeof.DOUBLE);
		DoubleCubedExample.insrs[6] = System.nanoTime();


		for(int i = 0; i < outputs.length; i++){
			collector.collect(outputs[i]);
		}

		DoubleCubedExample.insrs[7] = System.nanoTime();
		return null;
	}

	//@Override
	public void initialize(MutableObjectIterator<Double> input, Collector<Double> outputCollector) throws Exception{

		this.input = input;
		this.collector = outputCollector;

		ArrayList<Double> inputs = new ArrayList<>();
		Double next;
		while((next = input.next()) != null){
			inputs.add(next);
		}

		values = Doubles.toArray(inputs);

		this.size = inputs.size();
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
		cuModuleGetFunction(cubed, module, "double_cubed");

		// Create pointers
		pInputs = new CUdeviceptr();
		pOutputs = new CUdeviceptr();
		pSize = new CUdeviceptr();

		// Start Data Transfer

		// Allocating arrays for GPU
		cuMemAlloc(pInputs, this.size * Sizeof.DOUBLE);
		cuMemAlloc(pOutputs, this.size * Sizeof.DOUBLE);
		cuMemAlloc(pSize, Sizeof.INT);

		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		kernelParameters = Pointer.to(
			Pointer.to(pInputs),
			Pointer.to(pOutputs),
			Pointer.to(pSize)
		);
	}

	@Override
	public void releaseResources() {
		cuMemFree(pInputs);
		cuMemFree(pOutputs);
		cuMemFree(pSize);
	}

	@Override
	public void setDataProcessingTime(long time) {
		DoubleCubedExample.insrs[8] = time;
	}
}

