package org.apache.flink.examples.java.oldExamples;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.examples.java.goodExampes.GPUIsNumberPrime;
import org.apache.flink.examples.java.oldExamples.GPUMatrixInversion;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;


public class MatrixInversionExample {
	public static void main(String[] args) throws Exception{
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

		int[] numberOfNodesToTry = new int[]{16};
		int[] cpuCoefficients = new int[]{0};// 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,   1, 0};
		int[] gpuCoefficients = new int[]{1};// 1, 2, 3, 4, 7, 10, 15, 23, 35, 60, 130, 1};
		int times = 3;

		String resultLocation = "/home/skg113/gpuflink/results/30_huge_matrices_inverted.txt";
		FileWriter writer = new FileWriter(new File(resultLocation).getAbsoluteFile(), true);


		ArrayList<float[]> list = new ArrayList<>();

		Random random = new Random();

		for(int i = 0; i < 30; i++){
			float[] floats = new float[1000000];
			for (int j = 0; j < 1000000; j++) {
				floats[j] = random.nextFloat();
			}
			list.add(floats);
		}

		DataSet<float[]> matrices = env.fromCollection(list);


		String formatData = "Number Of Attempt, Percentages Given To The GPU, RAM to GPU, Kernel Call, GPU to RAM, End-to-End GPUMap, Total Time\n";
		printResult(formatData, writer);

		for (int numNodes : numberOfNodesToTry){
			for (int i = 0; i < gpuCoefficients.length; i++) {
				long accTime = 0;
				float gpuPercentage = (gpuCoefficients[i] * 100) / ((numNodes - 1) * cpuCoefficients[i] + gpuCoefficients[i]);
				for (int j = 0; j < times; j++) {


					DataSet<float[]> inverted = matrices.map(new GPUMatrixInversion()).setParallelism((cpuCoefficients[i] == 0) ? 1 : numNodes)
						.setCPUGPURatio(cpuCoefficients[i], gpuCoefficients[i]);
					try {
						inverted.collect();
					} catch (Exception e) {
						e.printStackTrace();
					}

					accTime = env.getLastJobExecutionResult().getNetRuntime();

					System.out.println("Runtime is : " + accTime);

					printResult(gpuPercentage+ "," + j + "," + accTime, writer);

					System.gc();
				}
			}
		}

	}

	private static void printResult(String data, FileWriter writer) throws Exception {
		writer.append(data);
		writer.append("\n");
		writer.flush();
	}
}
