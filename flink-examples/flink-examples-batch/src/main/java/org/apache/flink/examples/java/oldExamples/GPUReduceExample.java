package org.apache.flink.examples.java.oldExamples;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class GPUReduceExample {
	public static void main(String[] args) {
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

		ArrayList<Float> list = new ArrayList<>();

		Random random = new Random();

		for (long i = 0; i < 2000000; i++) {
			list.add(random.nextFloat());
		}

		DataSet<Float> floats = env.fromCollection(list);
		DataSet<Float> res = floats.reduce(new GPUReduceSum()).setParallelism(1);

		try {
			res.print();
		} catch (Exception e) {
			e.printStackTrace();
		}


		System.out.println("It took : " + env.getLastJobExecutionResult().getNetRuntime(TimeUnit.MILLISECONDS) + " ms");
	}
}
