import csv
from os import walk

def exploreFolder(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f in filenames:
            files.append(f.replace(".csv", ""))
        break
    return files


if __name__ == "__main__":

	files = exploreFolder('./')
	files.remove('.DS_Store')
	files.remove('train_process.py') #505

	for stock in files:
		rows = []
		ifile = open('./'+stock+'.csv', 'r')
		reader = csv.reader(ifile)
		for row in reader:
			rows.append(row)
		if len(rows)!=1000:
			print(stock)
			print(len(rows))

