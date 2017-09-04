import pandas as pd
import re
import csv
import random
cat_arr = ['Shopping', 'Business', 'Computers', 'Adult', 'Health', 'Society']
for cat in cat_arr:
	print (cat)
	df = pd.read_csv('dmoz0409_'+cat+'_train.csv', header=0)
	print (df.shape)
	count = 80000
	c = 120000

	with open('train_' + cat + '.csv', 'w',newline = '') as res:
		while c > 0 or count> 0:
			idx = random.randrange(1, 150000)
			if df['Category'][idx] == cat and count > 0:
				wr = csv.writer(res, dialect='excel')
				wr.writerow([df['URLofSite'][idx], df['Category'][idx]])
				count -= 1
			elif c > 0:
				wr = csv.writer(res, dialect='excel')
				wr.writerow([df['URLofSite'][idx], df['Category'][idx]])
				c -= 1
	res.close()