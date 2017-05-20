#!/usr/bin/env python

import sys
import os
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter

read_from = '/home/skg113/gpuflink/results'
write_to = '/home/skg113/gpuflink/graphs'

num_ratios = 10
num_repeated = 5

file_name = os.path.join(read_from, sys.argv[1])

f = open(file_name , 'r')

#Descriptions
f.readline()
f.readline()

raw_data = {'percentages': [],
        'host_to_device_time': [],
        'kernel_time': [],
        'device_to_host_time': [],
        'gpu_map_time': [],
        'whole_execution_time': []}

zipped = []

for i in xrange(num_ratios):
	percentage = 0
	host_to_device_times = []
	kernel_times = []
	device_to_host_times = []
	gpu_map_times = []
	whole_execution_times = []

	for j in xrange(num_repeated):
		tokens = f.readline().split(',')
		job_execution_time = float(tokens[6]) * 1000000
		end_to_end_time = float(tokens[5])
		percentage = float(tokens[0])		
		host_to_device_times.append(float(tokens[2])* 100 /end_to_end_time)
		kernel_times.append(float(tokens[3]) * 100 / end_to_end_time )
		device_to_host_times.append(float(tokens[4]) * 100 / end_to_end_time)
		gpu_map_times.append(float(tokens[5]) * 100 / job_execution_time)
		whole_execution_times.append(float(tokens[6]))
	if percentage == 0:
		continue
	raw_data['percentages'].append(percentage)
	raw_data['host_to_device_time'].append(statistics.mean(host_to_device_times))
	raw_data['kernel_time'].append(statistics.mean(kernel_times))
	raw_data['device_to_host_time'].append(statistics.mean(device_to_host_times))
	raw_data['gpu_map_time'].append(statistics.mean(gpu_map_times))
	raw_data['whole_execution_time'].append(statistics.mean(whole_execution_times))
	
df = pd.DataFrame(raw_data, columns = ['percentages', 'host_to_device_time', 'kernel_time', 'device_to_host_time', 'gpu_map_time'])

df

# Create the general blog and the "subplots" i.e. the bars
f, ax1 = plt.subplots(1, figsize=(10,5))

# Set the bar width
bar_width = 0.75

# positions of the left bar-boundaries
bar_l = [i+1 for i in range(len(df['host_to_device_time']))]

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/2) for i in bar_l]

# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the pre_score data
        df['gpu_map_time'],
        # set the width
        width=bar_width,
        # with the label pre score
        label='gpuMap call',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='blue')


# set the x ticks with names
plt.xticks(tick_pos, df['percentages'])

# Set the label and legends
ax1.set_ylabel("What percentage is the call to gpuMap \n out of the whole jobe execution time", fontsize=7)
ax1.set_xlabel("Percentage of the job ran on the gpu")
plt.legend(loc='upper left')
plt.title("20 million doubles cubed, Average of 5 runs, 16 nodes each")

# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

pp = PdfPages(os.path.join(write_to, sys.argv[1].split('.')[0] + '_gpuMap_vs_job_execution_graph.pdf'))

pp.savefig()

pp.close()

	
