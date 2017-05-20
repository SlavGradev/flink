#!/usr/bin/env python

import random
import sys

write_to = "~/gpuflink/data"

f = open(write_to + sys.argv[1], 'w')

n = sys.argv[2]

for i in xrange(n):
	f.write(str(random.random()))
	f.write("\n")

f.close()
	
