# Baseline, Better Baseline and Unet

To train the baseline model, get the predicted test masks and the csv file for submission run:

```bash
python baseline.py --data_dir <data_dir> --out_dir <output_directory>
```

To train the better baseline model, get the predicted test masks and the csv file for submission run:

```bash
python better_baseline.py --data_dir <data_dir> --out_dir <output_directory>
```

To train the Unet model, get the predicted test masks and the csv file for submission run:

```bash
python unet.py --data_dir <data_dir> --out_dir <output_directory>
```
