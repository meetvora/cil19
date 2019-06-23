submit=bsub -n 4 -W 24:00 -R "rusage[mem=18000, ngpus_excl_p=1]"

clear:
	rm ../data/exp/*

train:
	$(submit) python main.py

csv:
	cd ../scripts/
	python mask_to_submission.py
