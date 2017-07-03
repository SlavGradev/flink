package org.apache.flink.examples.java.oldExamples;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class GPUReduceExample {
	public static void main(String[] args) throws Exception {
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

		int[] numberOfNodesToTry = new int[]{16};
		int[] cpuCoefficients = new int[]{1, 0};// 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,   1, 0};
		int[] gpuCoefficients = new int[]{0, 1};// 1, 2, 3, 4, 7, 10, 15, 23, 35, 60, 130, 1};
		int times = 3;

		String resultLocation = "/home/skg113/gpuflink/results/20K_floats_added.txt";
		FileWriter writer = new FileWriter(new File(resultLocation).getAbsoluteFile(), true);


		ArrayList<Float> list = new ArrayList<>();

		Random random = new Random();

		for (long i = 0; i < 20000; i++) {
			list.add(random.nextFloat());
		}

		DataSet<Float> floatss = env.fromCollection(list);

		DataSet<Float> summ = floatss.reduce(new GPUReduceSum()).setParallelism(1).setGPUCoefficient(1);
		summ.print();

		float s = 0f;
		for (int i = 0; i < list.size(); i++) {
			s += list.get(i);
		}

		String formatData = "Number Of Attempt, Percentages Given To The GPU, RAM to GPU, Kernel Call, GPU to RAM, End-to-End GPUMap, Total Time\n";
		printResult(formatData, writer);

		for (int numNodes : numberOfNodesToTry){
			for (int i = 0; i < gpuCoefficients.length; i++) {
				long accTime = 0;
				float gpuPercentage = (gpuCoefficients[i] * 100) / ((numNodes - 1) * cpuCoefficients[i] + gpuCoefficients[i]);
				for (int j = 0; j < times; j++) {

					DataSet<Float> floats = env.fromCollection(list);

					DataSet<Float> sum = floats.reduce(new GPUReduceSum()).setParallelism((cpuCoefficients[i] == 0) ? 1 : numNodes)
						.setCPUGPURatio(cpuCoefficients[i], gpuCoefficients[i]);
					try {
						sum.collect();
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
