package org.apache.flink.examples.java.oldExamples;


import Jama.Matrix;
import jcuda.Pointer;
import jcuda.Sizeof;
import org.apache.flink.api.common.functions.GPUMapFunction;
import java.util.ArrayList;

import static jcuda.jcublas.JCublas.*;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

public class GPUMatrixInversion extends GPUMapFunction<float[], float[]> {

	@Override
	public float[] cpuMap(float[] value) {
		Matrix matrix = new Matrix(toDoubleArray(value));
		//matrix.print(100,5);
		Matrix inverted = matrix.inverse();
		//inverted.print(100, 5);
		return toFloatArray(inverted.getArray());
	}


	@Override
	public void initialize(int size) throws Exception {

	}

	@Override
	public float[][] gpuMap(ArrayList<float[]> values) {
		// Invert the matrix
		float[][] result = new float[values.size()][values.get(0).length];
		int i = 0;
		for(float[] value : values){
			float[] invA = value.clone();
			invertMatrix((int)Math.sqrt(value.length), invA);
			result[i] = invA;
			i += 1;
		}
		return values.toArray(result);
	}

	@Override
	public void releaseResources() {

	}

	@Override
	public void setDataProcessingTime(long time) {

	}

	public static void invertMatrix(int n, float A[])
	{
		Pointer dA = new Pointer();
		cublasAlloc(n * n, Sizeof.FLOAT, dA);
		cublasSetMatrix(n, n, Sizeof.FLOAT, Pointer.to(A), n, dA, n);

		invertMatrix(n, dA);

		cublasGetMatrix(n, n, Sizeof.FLOAT, dA, n, Pointer.to(A), n);
		cublasFree(dA);
	}

	public static void invertMatrix(int n, Pointer dA)
	{
		// Perform LU factorization
		int[] pivots = cudaSgetrfSquare(n, dA);

		// Perform inversion on factorized matrix
		cudaSgetri(n, dA, pivots);
	}

	private static int[] cudaSgetrfSquare(int n, Pointer dA)
	{
		int[] pivots = new int[n];
		for (int i = 0; i < n; i++)
		{
			pivots[i] = i;
		}

		float[] factor = { 0.0f };
		Pointer pFactor = Pointer.to(factor);
		for (int i = 0; i < n - 1; i++)
		{
			Pointer offset = at(dA, i * n + i);

			int pivot = i - 1 + cublasIsamax(n - i, offset, 1);
			if (pivot != i)
			{
				pivots[i] = pivot;
				cublasSswap(n, at(dA, pivot), n, at(dA, i), n);
			}

			cublasGetVector(1, Sizeof.FLOAT, offset, 1, pFactor, 1);
			cublasSscal(n - i - 1, 1 / factor[0], at(offset, 1), 1);
			cublasSger(n - i - 1, n - i - 1, -1.0f,
				at(offset, 1), 1, at(offset, n), n, at(offset, n + 1), n);
		}
		return pivots;
	}

	private static void cudaSgetri(int n, Pointer dA, int[] pivots)
	{
		// Perform inv(U)
		cudaStrtri(n, dA);

		// Solve inv(A)*L = inv(U)
		Pointer dWork = new Pointer();
		cublasAlloc(n - 1, Sizeof.FLOAT, dWork);

		for (int i = n - 1; i > 0; i--)
		{
			Pointer offset = at(dA, ((i - 1) * n + i));
			cudaMemcpy(dWork, offset, (n - 1) * Sizeof.FLOAT,
				cudaMemcpyDeviceToDevice);
			cublasSscal(n - i, 0, offset, 1);
			cublasSgemv('n', n, n - i, -1.0f,
				at(dA, i * n), n, dWork, 1, 1.0f, at(dA, ((i - 1) * n)), 1);
		}

		cublasFree(dWork);

		// Pivot back to original order
		for (int i = n - 1; i >= 0; i--)
		{
			if (i != pivots[i])
			{
				cublasSswap(n, at(dA, i * n), 1, at(dA, pivots[i] * n), 1);
			}
		}

	}

	private static Pointer at(Pointer p, int floatOffset)
	{
		return p.withByteOffset(floatOffset * Sizeof.FLOAT);
	}

	private static void cudaStrtri(int n, Pointer dA)
	{
		float[] factor = { 0.0f };
		Pointer pFactor = Pointer.to(factor);
		for (int i = 0; i < n; i++)
		{
			Pointer offset = at(dA, i * n);
			cublasGetVector(1, Sizeof.FLOAT, at(offset, i), 1, pFactor, 1);
			factor[0] = 1 / factor[0];
			cublasSetVector(1, Sizeof.FLOAT, pFactor, 1, at(offset, i), 1);
			cublasStrmv('u', 'n', 'n', i, dA, n, offset, 1);
			cublasSscal(i, -factor[0], offset, 1);
		}
	}

	private static float[] toFloatArray(double[][] array){
		float[] result = new float[array.length * array.length];
		int i = 0;
		for (int j = 0; j < array.length; j++) {
			for (int k = 0; k < array.length; k++) {
				result[i] = (float) array[j][k];
				i += 1;
			}
		}
		return result;
	}


	private double[][] toDoubleArray(float[] array) {
		int size = (int) Math.sqrt(array.length);
		double[][] result = new double[size][size];
		int i = 0;
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				result[j][k] = array[i];
				i += 1;
			}
		}
		return result;
	}

}
