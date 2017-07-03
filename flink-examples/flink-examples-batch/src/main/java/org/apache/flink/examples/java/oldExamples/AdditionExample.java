package org.apache.flink.examples.java.oldExamples;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.IterativeDataSet;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;


public class AdditionExample {
	public static long[] insrs = new long[]{0,0,0,0,0,0,0,0,0};
	public static void main(String[] args) throws Exception{

		int[] numberOfNodesToTry = new int[]{16};
		int[] gpuPercentages = new int[]{};
		int[] cpuCoefficients = new int[]{1};//{1, 1, 1, 1, 1,  1,  1,  1,  1,  1,   1, 0};
		int[] gpuCoefficients = new int[]{1};//{0, 1, 2, 4, 7, 10, 15, 23, 35, 60, 130, 1};
		int times = 2;

		String resultLocation = "/home/skg113/gpuflink/results/10_million_doubles_cubed.txt";
		File file = new File(resultLocation);
		FileWriter writer = new FileWriter(file.getAbsoluteFile(), true);

		HashMap<String, Long> resultq = new HashMap<>();

		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

		ArrayList<Integer> list = new ArrayList<>();

		for(int i = 0; i < 1000000; i++){
			list.add(i);
		}

		DataSet<Integer>  ints = env.fromCollection(list);


		DataSet<Integer> result = ints.map(new GPUAddition()).setParallelism(16).setGPUCoefficient(15);


		try {
			result.collect();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("It took : " + env.getLastJobExecutionResult().getNetRuntime(TimeUnit.MILLISECONDS) + " ms");
	}
}
