package org.apache.flink.api.common.functions;

import java.util.ArrayList;

public abstract class GPUSupportingMapFunction<T,O> implements MapFunction<T,O> {

	private boolean onGPU = false;
	protected byte[] params;

	@Override
	public O map(T value) throws Exception {
		return cpuMap(value);
	}

	public abstract O cpuMap(T value);
	public abstract O[] gpuMap(ArrayList<T> value);

	public void setOnGPU(boolean onGPU) {
		this.onGPU = onGPU;
	}

	public void setGPUParams(byte[] params) {
		this.params = params;
	}
}

