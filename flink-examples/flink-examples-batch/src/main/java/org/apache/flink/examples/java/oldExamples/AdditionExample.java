package org.apache.flink.examples.java;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;


public class AdditionExample {
	public static void main(String[] args){
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
		env.setParallelism(6);

		ArrayList<Integer> list = new ArrayList<>();

		for(int i = 0; i < 320000; i++){
			list.add(i);
		}


		DataSet<Integer> ints = env.fromCollection(list);
		DataSet<Integer> added = ints.map(new GPUAddition()).setParallelism(4).setGPUCoefficient(1);


		try {
			added.collect();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("It took : " + env.getLastJobExecutionResult().getNetRuntime(TimeUnit.MILLISECONDS) + " ms");
	}
}
