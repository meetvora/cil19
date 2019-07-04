submit=bsub -n 4 -W 24:00 -R "rusage[mem=18000, ngpus_excl_p=1]"

autoformat:
	yapf -ir --style pep8 src/

setup:
	pip install -r requirements.txt
	sh scripts/download_data.sh
	mkdir log

train:
	cd src && $(submit) python main.py

csv:
	python scripts/mask_to_submission.py $(EXP)
