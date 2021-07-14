## TransformerOCR for rerognizing handwriting text in Vietnamese

[TransformerOCR's structure](images/art.png)

This repository is inherited from the repository [https://github.com/fengxinjie/Transformer-OCR](https://github.com/fengxinjie/Transformer-OCR).

Link to the Vietnamese handwriting dataset: [https://github.com/hieunghia-pat/UIT-HWDB-dataset](https://github.com/hieunghia-pat/UIT-HWDB-dataset)

To evaluate the model, first download the chechpoint:

```bash
wget -d checkpoints
```

then adjust the path to the test dataset in [validation.py](validation.py) and run

```bash
python3 validation.py
```