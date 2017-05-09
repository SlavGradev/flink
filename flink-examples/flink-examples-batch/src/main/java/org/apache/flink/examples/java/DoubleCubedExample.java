package org.apache.flink.examples.java;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;


public class DoubleCubedExample {
	public static void main(String[] args){
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
		env.setParallelism(6);

		ArrayList<Double> list = new ArrayList<>();

		Random rand = new Random();

		for(int i = 0; i < 20000000; i++){
			list.add(rand.nextGaussian());
		}


		DataSet<Double> doubles = env.fromCollection(list);
		DataSet<Double> divided = doubles.map(new GPUDoubleCubed()).setParallelism(16).setGPUCoefficient(4);


		try {
			divided.collect();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("It took : " + env.getLastJobExecutionResult().getNetRuntime(TimeUnit.MILLISECONDS) + " ms");
	}
}
