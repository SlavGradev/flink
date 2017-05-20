package org.apache.flink.examples.java;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;

import java.io.*;
import java.util.*;
import java.util.concurrent.TimeUnit;


public class DoubleCubedExample {
	public static long[] insrs = new long[]{0,0,0,0,0,0,0,0,0};
	public static void main(String[] args) throws Exception{

		int[] numberOfNodesToTry = new int[]{16};
		int[] gpuPercentages = new int[]{};
		int[] cpuCoefficients = new int[]{1};//{1, 1, 1, 1, 1,  1,  1,  1,  1,  1, 1, 0};
		int[] gpuCoefficients = new int[]{15};//{0, 1, 2, 4, 7, 10, 15, 23, 35, 60, 130, 1};
		int times = 5;

		String datasetFileLocation = "/home/skg113/gpuflink/data/10_million_doubles.txt";
		String resultLocation = "/home/skg113/gpuflink/results/10_million_doubles_cubed.txt";
		File file = new File(resultLocation);
		FileWriter writer = new FileWriter(file.getAbsoluteFile(), true);

		if(cpuCoefficients.length != gpuCoefficients.length) {
			throw new RuntimeException("The coefficient arrays should be the same size");
		}

		HashMap<String, Long> result = new HashMap<>();
		ArrayList<Double> list = readDataSetFile(datasetFileLocation);
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

		String formatData = "Number Of Attempt, Percentages Given To The GPU, RAM to GPU, Kernel Call, GPU to RAM, End-to-End GPUMap, Total Time\n";
		printResult(formatData, writer);

		// Warm-up
		DataSet<Double> doubless = env.fromCollection(list);
		DataSet<Double> cubeds = doubless.map(new GPUDoubleCubed()).setParallelism(1)
			.setCPUGPURatio(0, 1);
		try {
			cubeds.collect();
		} catch (Exception e) {
			e.printStackTrace();
		}

		// Warm-up 2
		DataSet<Double> doublesss = env.fromCollection(list);
		DataSet<Double> cubedss = doubless.map(new GPUDoubleCubed()).setParallelism(16)
			.setCPUGPURatio(1, 1);
		try {
			cubeds.collect();
		} catch (Exception e) {
			e.printStackTrace();
		}

		for (int numNodes : numberOfNodesToTry){
			for (int i = 0; i < gpuCoefficients.length; i++) {
				long accTime = 0;
				float gpuPercentage = (gpuCoefficients[i] * 100) / ((numNodes - 1) * cpuCoefficients[i] + gpuCoefficients[i]);
				for (int j = 0; j < times; j++) {
					DataSet<Double> doubles = env.fromCollection(list);
					DataSet<Double> cubed = doubles.map(new GPUDoubleCubed()).setParallelism((cpuCoefficients[i] == 0) ? 1 : numNodes)
						.setCPUGPURatio(cpuCoefficients[i], gpuCoefficients[i]);
					try {
						cubed.collect();
					} catch (Exception e) {
						e.printStackTrace();
					}
					// Overhead
					long host_to_device_time = insrs[2] - insrs[1];
					long kernel_execution_time = insrs[4] - insrs[3];
					long device_to_host_time = insrs[6] - insrs[5];
					long gpuMap_time = insrs[7] - insrs[0];
					long data_processing_time = insrs[8] - gpuMap_time;

					accTime = env.getLastJobExecutionResult().getNetRuntime();

					printResult(gpuPercentage+ "," + j + "," + host_to_device_time + "," + kernel_execution_time + "," +
								device_to_host_time + "," + gpuMap_time + "," + data_processing_time + "," + accTime, writer);

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

	private static ArrayList<Double> readDataSetFile(String datasetFileLocation) {
		ArrayList<Double> result = new ArrayList<>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(datasetFileLocation));
			String line = br.readLine();

			while (line != null) {
				result.add(Double.valueOf(line));
				line = br.readLine();
			}

			br.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		 return result;
	}
}
