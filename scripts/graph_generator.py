#!/usr/bin/env python

import sys
import os
import statistics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter

read_from = '/home/skg113/gpuflink/results'
write_to = '/home/skg113/gpuflink/graphs'

num_ratios = 12
num_repeated = 2

file_name = os.path.join(read_from, sys.argv[1])

f = open(file_name , 'r')

#Descriptions
f.readline()
f.readline()

zipped = []

for i in xrange(num_ratios):
	times = []
	percentage = 0
	for j in xrange(num_repeated):
		tokens = f.readline().split(',')
		times.append(float(tokens[7]) * 10**-3)
		percentage = float(tokens[0])
	zipped.append((percentage, (statistics.mean(times), statistics.stdev(times))))
	
# Sort by percentage so that line can be straight
sortedData = sorted(zipped, key=itemgetter(0))
percentages, times_and_errs = zip(*sortedData)

times, errs = zip(*times_and_errs)

plt.plot(percentages, times, '-o', color='green')
plt.errorbar(percentages, times, errs, linestyle='None', ecolor='green')

plt.xlabel('Percentages of the Job given to a GPU', fontsize=12)
plt.ylabel('Time in Seconds', fontsize=12)
plt.title('Checking primality of 1536 32-bit prime numbers, Average of 2 runs, 16 nodes', fontsize=11)

pp = PdfPages(os.path.join(write_to, sys.argv[1].split('.')[0] + '_graph.pdf'))

pp.savefig()

pp.close()

	
