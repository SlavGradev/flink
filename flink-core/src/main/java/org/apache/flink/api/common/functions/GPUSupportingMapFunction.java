package org.apache.flink.api.common.functions;

import org.apache.flink.api.common.functions.MapFunction;

public abstract class GPUSupportingMapFunction<T,O> implements MapFunction<T,O> {

	private boolean onGPU = false;


	public void setOnGPU(boolean onGPU) {
		this.onGPU = onGPU;
	}

	@Override
	public O map(T value) throws Exception {
		if(onGPU){
			return gpuMap(value);
		}{
			return cpuMap(value);
		}
	}

	public abstract O cpuMap(T value);
	public abstract O gpuMap(T value);
}

