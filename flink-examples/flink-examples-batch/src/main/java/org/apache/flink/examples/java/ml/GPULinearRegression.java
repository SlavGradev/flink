/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.examples.java.ml;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.apache.flink.api.common.functions.*;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.examples.java.ml.util.LinearRegressionData;
import org.apache.flink.examples.java.ml.LinearRegression.Data;
import org.apache.flink.examples.java.ml.LinearRegression.Params;
import org.apache.flink.util.Collector;
import org.apache.flink.util.MutableObjectIterator;

import java.util.ArrayList;
import java.util.Collection;

import static jcuda.driver.JCudaDriver.*;

/**
 * This example implements a basic Linear Regression  to solve the y = theta0 + theta1*x problem using batch gradient descent algorithm.
 *
 * <p>
 * Linear Regression with BGD(batch gradient descent) algorithm is an iterative clustering algorithm and works as follows:<br>
 * Giving a data set and target set, the BGD try to find out the best parameters for the data set to fit the target set.
 * In each iteration, the algorithm computes the gradient of the cost function and use it to update all the parameters.
 * The algorithm terminates after a fixed number of iterations (as in this implementation)
 * With enough iteration, the algorithm can minimize the cost function and find the best parameters
 * This is the Wikipedia entry for the <a href = "http://en.wikipedia.org/wiki/Linear_regression">Linear regression</a> and <a href = "http://en.wikipedia.org/wiki/Gradient_descent">Gradient descent algorithm</a>.
 * 
 * <p>
 * This implementation works on one-dimensional data. And find the two-dimensional theta.<br>
 * It find the best Theta parameter to fit the target.
 * 
 * <p>
 * Input files are plain text files and must be formatted as follows:
 * <ul>
 * <li>Data points are represented as two double values separated by a blank character. The first one represent the X(the training data) and the second represent the Y(target).
 * Data points are separated by newline characters.<br>
 * For example <code>"-0.02 -0.04\n5.3 10.6\n"</code> gives two data points (x=-0.02, y=-0.04) and (x=5.3, y=10.6).
 * </ul>
 * 
 * <p>
 * This example shows how to use:
 * <ul>
 * <li> Bulk iterations
 * <li> Broadcast variables in bulk iterations
 * <li> Custom Java objects (PoJos)
 * </ul>
 */
@SuppressWarnings("serial")
public class GPULinearRegression {

	// *************************************************************************
	//     PROGRAM
	// *************************************************************************

	public static void main(String[] args) throws Exception {

		final ParameterTool params = ParameterTool.fromArgs(args);

		// set up execution environment
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
		env.setParallelism(1);
		final int iterations = params.getInt("iterations", 10);

		// make parameters available in the web interface
		env.getConfig().setGlobalJobParameters(params);

		// get input x data from elements
		DataSet<Data> data;
		if (params.has("input")) {
			String[] strings = new String[2];
			strings[0] = "x";
			strings[1] = "y";

			// read data from CSV file
			data = env.readCsvFile(params.get("input"))
					.fieldDelimiter(" ")
					.includeFields(true, true)
					.pojoType(Data.class, strings);
		} else {
			System.out.println("Executing LinearRegression example with default input data set.");
			System.out.println("Use --input to specify file input.");
			data = LinearRegressionData.getDefaultDataDataSet(env);
		}

		// get the parameters from elements
		DataSet<Params> parameters = LinearRegressionData.getDefaultParamsDataSet(env);

		// set number of bulk iterations for SGD linear Regression
		IterativeDataSet<Params> loop = parameters.iterate(iterations);

		DataSet<Params> new_parameters = data
			// compute a single step using every sample
			.map(new SubUpdate()).withBroadcastSet(loop, "parameters").setGPUCoefficient(1)
			// sum up all the steps
			.reduce(new UpdateAccumulator()).setGPUCoefficient(1)
			// average the steps and update all parameters
			.map(new Update()).setGPUCoefficient(1);

		// feed new parameters back into next iteration
		DataSet<Params> result = loop.closeWith(new_parameters);

		// emit result
		if(params.has("output")) {
			result.writeAsText(params.get("output"));
			// execute program
			env.execute("Linear Regression example");
		} else {
			System.out.println("Printing result to stdout. Use --output to specify output path: ");
			result.print();
			System.out.println(result.getExecutionEnvironment().getLastJobExecutionResult().getNetRuntime());
		}
	}

	// *************************************************************************
	//     DATA TYPES
	// *************************************************************************


	// *************************************************************************
	//     USER FUNCTIONS
	// *************************************************************************

	/**
	 * Compute a single BGD type update for every parameters.
	 */
	public static class SubUpdate extends GPURichMapFunction<Data,Tuple2<Params,Integer>> {

		private final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/output/LinearRegressionSubUpdate.ptx";

		private Collection<Params> parameters; 

		private Params parameter;

		private int count = 1;

		/** Reads the parameters from a broadcast variable into a collection. */
		@Override
		public void open(Configuration parameters) throws Exception {
			this.parameters = getRuntimeContext().getBroadcastVariable("parameters");
		}

		@Override
		public Tuple2<Params, Integer> cpuMap(Data value) {

			for(Params p : parameters){
				this.parameter = p;
			}

			double theta_0 = parameter.getTheta0() - 0.01*((parameter.getTheta0() + (parameter.getTheta1()*value.x)) - value.y);
			double theta_1 = parameter.getTheta1() - 0.01*(((parameter.getTheta0() + (parameter.getTheta1()*value.x)) - value.y) * value.x);

			return new Tuple2<Params,Integer>(new Params(theta_0,theta_1),count);
		}

		@Override
		public void initialize(int size) throws Exception {

		}

		@Override
		public Tuple2<Params, Integer>[] gpuMap(ArrayList<Data> values) {

			for(Params p : parameters){
				this.parameter = p;
			}


			if(values.size() == 0){
				return new Tuple2[0];
			}

			// Enable Exceptions
			JCudaDriver.setExceptionsEnabled(true);

			// Initialize the driver and create a context for the first device.
			cuInit(0);
			CUdevice device = new CUdevice();
			cuDeviceGet(device, 0);
			CUcontext context = new CUcontext();
			cuCtxCreate(context, 0, device);

			// Spawn a thread for each value
			int numThreads = values.size();

			// Load the ptx file.
			CUmodule module = new CUmodule();
			cuModuleLoad(module, moduleLocation);

			// Obtain a function pointer to the kernel function.
			CUfunction subUpdate = new CUfunction();
			cuModuleGetFunction(subUpdate, module, "linear_regression_sub_update");

			// Allocate t0, t1, xs and ys
			double[] t0 = new double[1];
			double[] t1 = new double[1];
			double[] xs = new double[values.size()];
			double[] ys = new double[values.size()];

			// Copy t0,t1,xs and ys
			t0[0] = parameter.getTheta0();
			t1[0] = parameter.getTheta1();
			for(int i = 0; i < values.size(); i++){
				xs[i] = values.get(i).x;
				ys[i] = values.get(i).y;
			}

			// Create pointers
			CUdeviceptr pt0s = new CUdeviceptr();
			CUdeviceptr pt1s = new CUdeviceptr();
			CUdeviceptr pxs = new CUdeviceptr();
			CUdeviceptr pys = new CUdeviceptr();
			CUdeviceptr pr0s = new CUdeviceptr();
			CUdeviceptr pr1s = new CUdeviceptr();
			CUdeviceptr pN = new CUdeviceptr();

			// Allocating arrays for GPU
			cuMemAlloc(pt0s, Sizeof.DOUBLE);
			cuMemAlloc(pt1s, Sizeof.DOUBLE);
			cuMemAlloc(pxs, numThreads * Sizeof.DOUBLE);
			cuMemAlloc(pys, numThreads * Sizeof.DOUBLE);
			cuMemAlloc(pr0s, numThreads * Sizeof.DOUBLE);
			cuMemAlloc(pr1s, numThreads * Sizeof.DOUBLE);
			cuMemAlloc(pN, Sizeof.INT);

			// Copy over t0, t1, xs, and ys to device memory
			cuMemcpyHtoD(pt0s, Pointer.to(t0), Sizeof.DOUBLE);
			cuMemcpyHtoD(pt1s, Pointer.to(t1), Sizeof.DOUBLE);
			cuMemcpyHtoD(pxs, Pointer.to(xs), numThreads * Sizeof.DOUBLE);
			cuMemcpyHtoD(pys, Pointer.to(ys), numThreads * Sizeof.DOUBLE);
			cuMemcpyHtoD(pN , Pointer.to(new int[] {values.size()}), Sizeof.INT);

			// Set up the kernel parameters: A pointer to an array
			// of pointers which point to the actual values.
			Pointer kernelParameters = Pointer.to(
				Pointer.to(pt0s),
				Pointer.to(pt1s),
				Pointer.to(pxs),
				Pointer.to(pys),
				Pointer.to(pr0s),
				Pointer.to(pr1s),
				Pointer.to(pN)
			);

			int blockSize = 512;
			// Call the kernel function.
			cuLaunchKernel(subUpdate,
				(int)Math.ceil( (double) values.size() / blockSize),  1, 1,      // Grid dimension
				blockSize, 1, 1,      // Block dimension
				0, null,               // Shared memory size and stream
				kernelParameters, null // Kernel- and extra parameters
			);

			cuCtxSynchronize();

			// Store the output in r0 and r1
			double[] r0s = new double[values.size()];
			double[] r1s = new double[values.size()];

			// Copy from device memory to host memory
			cuMemcpyDtoH(Pointer.to(r0s), pr0s, numThreads * Sizeof.DOUBLE);
			cuMemcpyDtoH(Pointer.to(r1s), pr1s, numThreads * Sizeof.DOUBLE);

			Tuple2<Params, Integer>[] result = new Tuple2[values.size()];

			for(int i = 0; i < values.size(); i++){
				result[i] = new Tuple2<Params, Integer>(new Params(r0s[i], r1s[i]), count);
			}

			cuMemFree(pt0s);
			cuMemFree(pt1s);
			cuMemFree(pxs);
			cuMemFree(pys);
			cuMemFree(pr0s);
			cuMemFree(pr1s);
			cuMemFree(pN);

			return result;

		}

		@Override
		public void releaseResources() {

		}

		@Override
		public void setDataProcessingTime(long time) {

		}


	}

	/**  
	 * Accumulator all the update.
	 * */
	public static class UpdateAccumulator extends GPUReduceFunction<Tuple2<Params,Integer>> {

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
		public Tuple2<Params, Integer> reduce(Tuple2<Params, Integer> val1, Tuple2<Params, Integer> val2) {

			double new_theta0 = val1.f0.getTheta0() + val2.f0.getTheta0();
			double new_theta1 = val1.f0.getTheta1() + val2.f0.getTheta1();
			Params result = new Params(new_theta0,new_theta1);
			return new Tuple2<Params, Integer>( result, val1.f1 + val2.f1);

		}

		private static final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/output/LinearRegressionUpdateAccumulator.ptx";

		@Override
		public Tuple2<Params, Integer> reduce(ArrayList<Tuple2<Params, Integer>> values) throws Exception {

			if(values.size() == 0){
				return new Tuple2();
			}



			double[] t0s = new double[values.size()];
			double[] t1s = new double[values.size()];

			init();
			return new Tuple2<Params, Integer>(new Params(reduce(t0s), reduce(t1s)), values.size());

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
			cuMemAlloc(deviceBuffer, 1024 * Sizeof.DOUBLE);

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
		public static double reduce(double hostInput[])
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
		public static double reduce(
			double hostInput[], int maxThreads, int maxBlocks)
		{
			// Allocate and fill the device memory
			CUdeviceptr deviceInput = new CUdeviceptr();
			cuMemAlloc(deviceInput, hostInput.length * Sizeof.FLOAT);
			cuMemcpyHtoD(deviceInput, Pointer.to(hostInput),
				hostInput.length * Sizeof.FLOAT);

			// Call reduction on the device memory
			double result =
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
		public static double reduce(
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
		public static double reduce(
			Pointer deviceInput, int numElements,
			int maxThreads, int maxBlocks)
		{
			// Determine the number of threads and blocks
			// (as done in the NVIDIA sample)
			int numBlocks = getNumBlocks(numElements, maxBlocks, maxThreads);
			int numThreads = getNumThreads(numElements, maxBlocks, maxThreads);

			// Call the main reduction method
			double result = reduce(numElements, numThreads, numBlocks,
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
		private static double reduce(
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
			cuMemcpyDtoH(Pointer.to(result), deviceBuffer, Sizeof.DOUBLE);
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
			int sharedMemSize = threads * Sizeof.DOUBLE;
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


	}

	/**
	 * Compute the final update by average them.home
	 */
	public static class Update extends GPUMapFunction<Tuple2<Params,Integer>,Params> {

		private final String moduleLocation = "/home/skg113/gpuflink/gpuflink-kernels/output/LinearRegressionUpdate.ptx";

		@Override
		public Params cpuMap(Tuple2<Params, Integer> value) {
			return value.f0.div(value.f1);
		}

		@Override
		public Params[] gpuMap(ArrayList<Tuple2<Params, Integer>> values) {
			if(values.size() == 0){
				return new Params[0];
			}
			System.out.println("GPGPU");
			// Enable Exceptions
			JCudaDriver.setExceptionsEnabled(true);

			// Initialize the driver and create a context for the first device.
			cuInit(0);
			CUdevice device = new CUdevice();
			cuDeviceGet(device, 0);
			CUcontext context = new CUcontext();
			cuCtxCreate(context, 0, device);

			// Spawn a thread for each value
			int numThreads = values.size();

			// Load the ptx file.
			CUmodule module = new CUmodule();
			cuModuleLoad(module, moduleLocation);

			// Obtain a function pointer to the kernel function.
			CUfunction update = new CUfunction();
			cuModuleGetFunction(update, module, "linear_regression_update");

			// Allocate t0, t1 and c
			double[] t0s = new double[values.size()];
			double[] t1s = new double[values.size()];
			int[] cs = new int[values.size()];

			// Copy t0, t1 and c
			for(int i = 0; i < values.size(); i++){
				t0s[i] = values.get(i).f0.getTheta0();
				t1s[i] = values.get(i).f0.getTheta1();
				cs[i] = values.get(i).f1;
			}

			// Create pointers
			CUdeviceptr pt0s = new CUdeviceptr();
			CUdeviceptr pt1s = new CUdeviceptr();
			CUdeviceptr pr0s = new CUdeviceptr();
			CUdeviceptr pr1s = new CUdeviceptr();
			CUdeviceptr pcs = new CUdeviceptr();
			CUdeviceptr pN = new CUdeviceptr();

			// Allocating arrays for GPU
			cuMemAlloc(pt0s, numThreads * Sizeof.DOUBLE);
			cuMemAlloc(pt1s, numThreads * Sizeof.DOUBLE);
			cuMemAlloc(pr0s, numThreads * Sizeof.DOUBLE);
			cuMemAlloc(pr1s, numThreads * Sizeof.DOUBLE);
			cuMemAlloc(pcs, numThreads * Sizeof.INT);
			cuMemAlloc(pN, Sizeof.INT);

			// Copy over t0, t1 and c to device memory
			cuMemcpyHtoD(pt0s, Pointer.to(t0s), numThreads * Sizeof.DOUBLE);
			cuMemcpyHtoD(pt1s, Pointer.to(t1s), numThreads * Sizeof.DOUBLE);
			cuMemcpyHtoD(pN, Pointer.to(new int[]{values.size()}), Sizeof.INT);



			// Set up the kernel parameters: A pointer to an array
			// of pointers which point to the actual values.
			Pointer kernelParameters = Pointer.to(
				Pointer.to(pt0s),
				Pointer.to(pt1s),
				Pointer.to(pr0s),
				Pointer.to(pr1s),
				Pointer.to(pcs),
				Pointer.to(pN)
			);

			int blockSize = 512;

			// Call the kernel function.
			cuLaunchKernel(update,
				(int)Math.ceil( (double) values.size() / blockSize),  1, 1,      // Grid dimension
				blockSize, 1, 1,      // Block dimension
				0, null,               // Shared memory size and stream
				kernelParameters, null // Kernel- and extra parameters
			);

			cuCtxSynchronize();

			// Store the output in r0 and r1
			double[] r0s = new double[values.size()];
			double[] r1s = new double[values.size()];

			// Copy from device memory to host memory
			cuMemcpyDtoH(Pointer.to(r0s), pr0s, numThreads * Sizeof.DOUBLE);
			cuMemcpyDtoH(Pointer.to(r1s), pr1s, numThreads * Sizeof.DOUBLE);

			Params[] result = new Params[values.size()];

			for(int i = 0; i < values.size(); i++){
				result[i] = new Params(r0s[i], r1s[i]);
			}

			cuMemFree(pt0s);
			cuMemFree(pt1s);
			cuMemFree(pr0s);
			cuMemFree(pr1s);
			cuMemFree(pcs);
			cuMemFree(pN);

			return result;
		}

		@Override
		public void initialize(int size) throws Exception {

		}

		@Override
		public void releaseResources() {

		}

		@Override
		public void setDataProcessingTime(long time) {

		}


	}

}

