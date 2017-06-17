package org.apache.flink.examples.java.oldExamples;

import com.google.common.primitives.Floats;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.flink.api.common.functions.GPUReduceFunction;

import java.util.ArrayList;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class GPUReduceSum extends GPUReduceFunction<Float> {

	private static final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/output/SumReduce.ptx";


	/**
	 * The CUDA context created by this sample
	 */
	private static CUcontext context;

	/**
	 * The module which is loaded in form of a PTX file
	 */
	private static CUmodule module;

	/**
	 * The actual kernel function from the module
	 */
	private static CUfunction function;

	/**
	 * Temporary memory for the device output
	 */
	private static CUdeviceptr deviceBuffer;

	@Override
	public Float reduce(ArrayList<Float> values) throws Exception {
		init();
		return reduce(Floats.toArray(values));
	}

	/**
	 * Initialize the driver API and create a context for the first
	 * device, and then call {@link #prepare()}
	 */
	private static void init()
	{
		cuInit(0);
		CUdevice device = new CUdevice();
		cuDeviceGet(device, 0);
		context = new CUcontext();
		cuCtxCreate(context, 0, device);
		prepare();
	}

	/**
	 * Prepare everything for calling the reduction kernel function.
	 * This method assumes that a context already has been created
	 * and is current!
	 */
	public static void prepare()
	{
		// Prepare the ptx file.
		String ptxFileName = moduleLocation;


		// Load the module from the PTX file
		module = new CUmodule();
		cuModuleLoad(module, ptxFileName);

		// Obtain a function pointer to the "reduce" function.
		function = new CUfunction();
		cuModuleGetFunction(function, module, "reduce");

		// Allocate a chunk of temporary memory (must be at least
		// numberOfBlocks * Sizeof.FLOAT)
		deviceBuffer = new CUdeviceptr();
		cuMemAlloc(deviceBuffer, 1024 * Sizeof.FLOAT);

	}

	/**
	 * Release all resources allocated by this class
	 */
	public static void shutdown()
	{
		cuModuleUnload(module);
		cuMemFree(deviceBuffer);
		if (context != null)
		{
			cuCtxDestroy(context);
		}
	}

	/**
	 * Perform a reduction on the given input, with a default number
	 * of threads and blocks, and return the result. <br />
	 * <br />
	 * This method assumes that either {@link #init()} or
	 * {@link #prepare()} have already been called.
	 *
	 * @param hostInput The input to reduce
	 * @return The reduction result
	 */
	public static float reduce(float hostInput[])
	{
		return reduce(hostInput, 128, 64);
	}

	/**
	 * Perform a reduction on the given input, with the given number
	 * of threads and blocks, and return the result. <br />
	 * <br />
	 * This method assumes that either {@link #init()} or
	 * {@link #prepare()} have already been called.
	 *
	 * @param hostInput The input to reduce
	 * @param maxThreads The maximum number of threads per block
	 * @param maxBlocks The maximum number of blocks per grid
	 * @return The reduction result
	 */
	public static float reduce(
		float hostInput[], int maxThreads, int maxBlocks)
	{
		// Allocate and fill the device memory
		CUdeviceptr deviceInput = new CUdeviceptr();
		cuMemAlloc(deviceInput, hostInput.length * Sizeof.FLOAT);
		cuMemcpyHtoD(deviceInput, Pointer.to(hostInput),
			hostInput.length * Sizeof.FLOAT);

		// Call reduction on the device memory
		float result =
			reduce(deviceInput, hostInput.length, maxThreads, maxBlocks);

		// Clean up and return the result
		cuMemFree(deviceInput);
		return result;
	}


	/**
	 * Performs a reduction on the given device memory with the given
	 * number of elements.
	 *
	 * @param deviceInput The device input memory
	 * @param numElements The number of elements to reduce
	 * @return The reduction result
	 */
	public static float reduce(
		Pointer deviceInput, int numElements)
	{
		return reduce(deviceInput, numElements, 128, 64);
	}


	/**
	 * Performs a reduction on the given device memory with the given
	 * number of elements and the specified limits for threads and
	 * blocks.
	 *
	 * @param deviceInput The device input memory
	 * @param numElements The number of elements to reduce
	 * @param maxThreads The maximum number of threads
	 * @param maxBlocks The maximum number of blocks
	 * @return The reduction result
	 */
	public static float reduce(
		Pointer deviceInput, int numElements,
		int maxThreads, int maxBlocks)
	{
		// Determine the number of threads and blocks
		// (as done in the NVIDIA sample)
		int numBlocks = getNumBlocks(numElements, maxBlocks, maxThreads);
		int numThreads = getNumThreads(numElements, maxBlocks, maxThreads);

		// Call the main reduction method
		float result = reduce(numElements, numThreads, numBlocks,
			maxThreads, maxBlocks, deviceInput);
		return result;
	}



	/**
	 * Performs a reduction on the given device memory.
	 *
	 * @param n The number of elements for the reduction
	 * @param numThreads The number of threads
	 * @param numBlocks The number of blocks
	 * @param maxThreads The maximum number of threads
	 * @param maxBlocks The maximum number of blocks
	 * @param deviceInput The input memory
	 * @return The reduction result
	 */
	private static float reduce(
		int  n, int  numThreads, int  numBlocks,
		int  maxThreads, int  maxBlocks, Pointer deviceInput)
	{
		// Perform a "tree like" reduction as in the NVIDIA sample
		reduce(n, numThreads, numBlocks, deviceInput, deviceBuffer);
		int s=numBlocks;
		while(s > 1)
		{
			int threads = getNumThreads(s, maxBlocks, maxThreads);
			int blocks = getNumBlocks(s, maxBlocks, maxThreads);

			reduce(s, threads, blocks, deviceBuffer, deviceBuffer);
			s = (s + (threads*2-1)) / (threads*2);
		}

		float result[] = {0.0f};
		cuMemcpyDtoH(Pointer.to(result), deviceBuffer, Sizeof.FLOAT);
		return result[0];
	}


	/**
	 * Perform a reduction of the specified number of elements in the given
	 * device input memory, using the given number of threads and blocks,
	 * and write the results into the given output memory.
	 *
	 * @param size The size (number of elements)
	 * @param threads The number of threads
	 * @param blocks The number of blocks
	 * @param deviceInput The device input memory
	 * @param deviceOutput The device output memory. Its size must at least
	 * be numBlocks*Sizeof.FLOAT
	 */
	private static void reduce(int size, int threads, int blocks,
							   Pointer deviceInput, Pointer deviceOutput)
	{
		//System.out.println("Reduce "+size+" elements with "+
		//    threads+" threads in "+blocks+" blocks");

		// Compute the shared memory size (as done in
		// the NIVIDA sample)
		int sharedMemSize = threads * Sizeof.FLOAT;
		if (threads <= 32)
		{
			sharedMemSize *= 2;
		}

		// Set up the kernel parameters: A pointer to an array
		// of pointers which point to the actual values.
		Pointer kernelParameters = Pointer.to(
			Pointer.to(deviceInput),
			Pointer.to(deviceOutput),
			Pointer.to(new int[]{size})
		);

		// Call the kernel function.
		cuLaunchKernel(function,
			blocks,  1, 1,         // Grid dimension
			threads, 1, 1,         // Block dimension
			sharedMemSize, null,   // Shared memory size and stream
			kernelParameters, null // Kernel- and extra parameters
		);
		cuCtxSynchronize();
	}


	/**
	 * Compute the number of blocks that should be used for the
	 * given input size and limits
	 *
	 * @param n The input size
	 * @param maxBlocks The maximum number of blocks
	 * @param maxThreads The maximum number of threads
	 * @return The number of blocks
	 */
	private static int getNumBlocks(int n, int maxBlocks, int maxThreads)
	{
		int blocks = 0;
		int threads = getNumThreads(n, maxBlocks, maxThreads);
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
		blocks = Math.min(maxBlocks, blocks);
		return blocks;
	}

	/**
	 * Compute the number of threads that should be used for the
	 * given input size and limits
	 *
	 * @param n The input size
	 * @param maxBlocks The maximum number of blocks
	 * @param maxThreads The maximum number of threads
	 * @return The number of threads
	 */
	private static int getNumThreads(int n, int maxBlocks, int maxThreads)
	{
		int threads = 0;
		threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
		return threads;
	}

	/**
	 * Returns the power of 2 that is equal to or greater than x
	 *
	 * @param x The input
	 * @return The next power of 2
	 */
	private static int nextPow2(int x)
	{
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}


	@Override
	public Float reduce(Float value1, Float value2) throws Exception {
		return value1 + value2;
	}
}
