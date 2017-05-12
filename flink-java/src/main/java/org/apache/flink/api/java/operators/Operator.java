/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.api.java.operators;

import org.apache.flink.annotation.Public;
import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.util.Preconditions;

/**
 * Base class of all operators in the Java API.
 * 
 * @param <OUT> The type of the data set produced by this operator.
 * @param <O> The type of the operator, so that we can return it.
 */
@Public
public abstract class Operator<OUT, O extends Operator<OUT, O>> extends DataSet<OUT> {

	protected String name;
	
	protected int parallelism = ExecutionConfig.PARALLELISM_DEFAULT;
	private int gpuCoefficient = ExecutionConfig.GPU_COEFFICIENT_DEFAULT;
	private int cpuCoefficient = ExecutionConfig.CPU_COEFFICIENT_DEFAULT;

	protected Operator(ExecutionEnvironment context, TypeInformation<OUT> resultType) {
		super(context, resultType);
	}
	
	/**
	 * Returns the type of the result of this operator.
	 * 
	 * @return The result type of the operator.
	 */
	public TypeInformation<OUT> getResultType() {
		return getType();
	}

	/**
	 * Returns the name of the operator. If no name has been set, it returns the name of the
	 * operation, or the name of the class implementing the function of this operator.
	 * 
	 * @return The name of the operator.
	 */
	public String getName() {
		return name;
	}
	
	/**
	 * Returns the parallelism of this operator.
	 * 
	 * @return The parallelism of this operator.
	 */
	public int getParallelism() {
		return this.parallelism;
	}

	/**
	 * Returns the gpu coefficient of this operator.
	 *
	 * @return The gpu coefficient of this operator.
	 */
	public int getGpuCoefficient() {
		return this.gpuCoefficient;
	}

	/**
	 * Returns the cpu coefficient of this operator.
	 *
	 * @return The cpu coefficient of this operator.
	 */
	public int getCpuCoefficient() {
		return this.cpuCoefficient;
	}

	/**
	 * Sets the name of this operator. This overrides the default name, which is either
	 * a generated description of the operation (such as for example "Aggregate(1:SUM, 2:MIN)")
	 * or the name the user-defined function or input/output format executed by the operator.
	 * 
	 * @param newName The name for this operator.
	 * @return The operator with a new name.
	 */
	public O name(String newName) {
		this.name = newName;
		@SuppressWarnings("unchecked")
		O returnType = (O) this;
		return returnType;
	}
	
	/**
	 * Sets the parallelism for this operator.
	 * The parallelism must be 1 or more.
	 * 
	 * @param parallelism The parallelism for this operator. A value equal to {@link ExecutionConfig#PARALLELISM_DEFAULT}
	 *        will use the system default.
	 * @return The operator with set parallelism.
	 */
	public O setParallelism(int parallelism) {
		Preconditions.checkArgument(parallelism > 0 || parallelism == ExecutionConfig.PARALLELISM_DEFAULT,
			"The parallelism of an operator must be at least 1.");

		this.parallelism = parallelism;

		@SuppressWarnings("unchecked")
		O returnType = (O) this;
		return returnType;
	}

	/**
	 * Sets the gpuCoefficient for this operator.
	 * The coefficient must be 0 or more.
	 *
	 * @param gpuCoefficient The coefficient for this operator. A value equal to {@link ExecutionConfig#GPU_COEFFICIENT_DEFAULT}
	 *        will use the system default.
	 * @return The operator with set gpu coefficient.
	 */
	public O setGPUCoefficient(int gpuCoefficient) {
		Preconditions.checkArgument(gpuCoefficient >= 0,
			"The gpu coefficient of an operator must be at least 0.");

		this.gpuCoefficient = gpuCoefficient;

		@SuppressWarnings("unchecked")
		O returnType = (O) this;
		return returnType;
	}

	/**
	 * Sets the gpuRation for this operator.
	 * The gpu coefficient must be 0 or more.
	 * The cpu coefficient must be 1 or more.
	 *
	 * @param gpuCoefficient The coefficient for this operator. A value equal to {@link ExecutionConfig#GPU_COEFFICIENT_DEFAULT}
	 *        will use the system default.
	 * @param cpuCoefficient The coefficient for this operator. A value equal to {@link ExecutionConfig#CPU_COEFFICIENT_DEFAULT}
	 *        will use the system default.
	 * @return The operator with set gpu coefficient.
	 */
	public O setCPUGPURatio(int cpuCoefficient, int gpuCoefficient) {
		Preconditions.checkArgument(gpuCoefficient >= 0,
			"The gpu coefficient of an operator must be at least 0.");

		this.cpuCoefficient = cpuCoefficient;
		this.gpuCoefficient = gpuCoefficient;

		@SuppressWarnings("unchecked")
		O returnType = (O) this;
		return returnType;
	}
}
