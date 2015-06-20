train.dat: train.txt
	perl -pe 's/\((.*)\)/$$1/' < train.txt > train.dat
