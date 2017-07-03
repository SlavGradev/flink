package org.apache.flink.examples.java.goodExampes;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

public class IsNumberPrimeExample {


	public static long[] insrs = new long[]{0,0,0,0,0,0,0,0,0};
	public static void main(String[] args) throws Exception{

		int[] numberOfNodesToTry = new int[]{16};
		int[] gpuPercentages = new int[]{};
		int[] cpuCoefficients = new int[]{0};//{0,   1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1};//{1, 1, 1, 1, 1,  1,  1,  1,  1,  1,   1, 0};
		int[] gpuCoefficients = new int[]{1};//{1, 130, 60, 35, 23, 15, 10, 7, 4, 2, 1, 0};//{0, 1, 2, 4, 7, 10, 15, 23, 35, 60, 130, 1};
		int times = 1;

		String datasetFileLocation = "/home/skg113/gpuflink/data/2000_primes.txt";
		String resultLocation = "/home/skg113/gpuflink/results/2000_primes.txt";
		File file = new File(resultLocation);
		FileWriter writer = new FileWriter(file.getAbsoluteFile(), true);

		if(cpuCoefficients.length != gpuCoefficients.length) {
			throw new RuntimeException("The coefficient arrays should be the same size");
		}

		HashMap<String, Long> result = new HashMap<>();
		ArrayList<Integer> list = readDataSetFile(datasetFileLocation);
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();


		// Warm-up
		DataSet<Integer> doubless = env.fromCollection(list);
		DataSet<Boolean> cubeds = doubless.map(new MapFunction<Integer, Boolean>() {
			@Override
			public Boolean map(Integer value) throws Exception {
				return true;
			}
		}).setParallelism(1);
		try {
			cubeds.print();
		} catch (Exception e) {
			e.printStackTrace();
		}
		long acTime = env.getLastJobExecutionResult().getNetRuntime(TimeUnit.NANOSECONDS);

		System.out.println("Runtime is : " + acTime);


		// Warm-up 2
		DataSet<Integer> doublesss = env.fromCollection(list);
		DataSet<Boolean> cubedss = doublesss.map(new MapFunction<Integer, Boolean>() {
			@Override
			public Boolean map(Integer value) throws Exception {
				return true;
			}
		}).setParallelism(2);
		try {
			cubedss.print();
		} catch (Exception e) {
			e.printStackTrace();
		}

		acTime = env.getLastJobExecutionResult().getNetRuntime();

		System.out.println("Runtime is : " + acTime);


		String formatData = "Number Of Attempt, Percentages Given To The GPU, RAM to GPU, Kernel Call, GPU to RAM, End-to-End GPUMap, Total Time\n";
		printResult(formatData, writer);

		for (int numNodes : numberOfNodesToTry){
			for (int i = 0; i < gpuCoefficients.length; i++) {
				long accTime = 0;
				float gpuPercentage = (gpuCoefficients[i] * 100) / ((numNodes - 1) * cpuCoefficients[i] + gpuCoefficients[i]);
				for (int j = 0; j < times; j++) {
					DataSet<Integer> doubles = env.fromCollection(list);
					DataSet<Boolean> cubed = doubles.map(new GPUIsNumberPrime()).setParallelism((cpuCoefficients[i] == 0) ? 1 : numNodes)
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
					long data_processing_time = insrs[8];

					accTime = env.getLastJobExecutionResult().getNetRuntime();

					System.out.println("Runtime is : " + accTime);


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

	private static ArrayList<Integer> readDataSetFile(String datasetFileLocation) {
		ArrayList<Integer> result = new ArrayList<>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(datasetFileLocation));
			String line = br.readLine();

			while (line != null) {
				result.add(Integer.valueOf(line));
				line = br.readLine();
			}

			br.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		return result;
	}


}
