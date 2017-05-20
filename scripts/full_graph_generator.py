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
num_repeated = 2

millisecond_in_nanoseconds = 1000000
nanosecond_in_milliseconds = 1 / 1000000

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
		percentage = float(tokens[0])		
		host_to_device_times.append(float(tokens[2]))
		kernel_times.append(float(tokens[3]))
		device_to_host_times.append(float(tokens[4]))
		gpu_map_times.append(float(tokens[5]))
		whole_execution_times.append(float(tokens[6]) * millisecond_in_nanoseconds)
	if percentage == 0:
		continue
	raw_data['percentages'].append(percentage)
	raw_data['host_to_device_time'].append(statistics.mean(host_to_device_times))
	raw_data['kernel_time'].append(statistics.mean(kernel_times))
	raw_data['device_to_host_time'].append(statistics.mean(device_to_host_times))
	raw_data['gpu_map_time'].append(statistics.mean(gpu_map_times))
	raw_data['whole_execution_time'].append(statistics.mean(whole_execution_times))
	
df = pd.DataFrame(raw_data, columns = ['percentages', 'host_to_device_time', 'kernel_time', 'device_to_host_time', 'gpu_map_time', 'whole_execution_time'])

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
        df['whole_execution_time'],
        # set the width
        width=bar_width,
        # with the label pre score
        label='Whole Execution Time',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='cyan')


# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the pre_score data
        df['gpu_map_time'],
        # set the width
        width=bar_width,
        # with the label pre score
        label='Call to gpuMap time',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='red')

# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the pre_score data
        df['host_to_device_time'],
        # set the width
        width=bar_width,
        # with the label pre score
        label='host_to_device_transfer_time',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='blue')

# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the kernel_time data
        df['kernel_time'],
        # set the width
        width=bar_width,
        # with pre_score on the bottom
        bottom=df['host_to_device_time'],
        # with the label mid score
        label='kernel_execution_time',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='green')

# Create a bar plot, in position bar_1
ax1.bar(bar_l,
        # using the post_score data
        df['device_to_host_time'],
        # set the width
        width=bar_width,
        # with pre_score and kernel_time on the bottom
        bottom=[i+j for i,j in zip(df['kernel_time'],df['host_to_device_time'])],
        # with the label post score
        label='device_to_host_transfer_time',
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='yellow')

# set the x ticks with names
plt.xticks(tick_pos, df['percentages'])

# Set the label and legends
ax1.set_ylabel("Time in Nanoseconds", fontsize=10)
ax1.set_xlabel("Percentage of the job ran on the gpu")

ax1.set_yscale('log')

plt.legend(loc='upper left')
plt.title("10 million doubles cubed, Average of 2 runs, 16 nodes each")

# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

pp = PdfPages(os.path.join(write_to, sys.argv[1].split('.')[0] + '_full_graph.pdf'))

pp.savefig()

pp.close()

	
