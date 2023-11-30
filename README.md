# AI Hematologist

## Overview
Microscopic examination and quantification of blood cells is a pivotal cornerstones of hematological diagnostics.
This task, tradiitonal performed manually by hematologists, involves annotating blood cells, which can be both time-consuming and monotonous.

### Solution
Two solutions are provided within this package including
- A DeepLab CNN which segments the entire blood cell
- A Vit which classifies each pixel as either nucleas or background through a sliding window approach

In `src.utils` you can find additional functions to combine the results from both architectures.

### Usage

The blood cell segmentation model can be trained and evaluated as follows:
```commandline
python src/train_cnn_cell.py [arguments]
```

Descriptions of `[arguments]` can be found with the `--help` flag. 

Similar the nucleas segmentation model can be trained as follows:
```commandline
python src/train_vit_nucleas.py [arguments]
```
