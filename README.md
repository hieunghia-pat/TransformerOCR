## TransformerOCR for rerognizing handwriting text in Vietnamese

[TransformerOCR's structure](images/art.png)

This repository is inherited from the repository [https://github.com/fengxinjie/Transformer-OCR](https://github.com/fengxinjie/Transformer-OCR).

Link to the Vietnamese handwriting dataset: [https://github.com/hieunghia-pat/UIT-HWDB-dataset](https://github.com/hieunghia-pat/UIT-HWDB-dataset)

To evaluate the model, first download the chechpoint:

```bash
wget https://www.googleapis.com/drive/v3/files/1ApGhHBpce5dkGwoZdzGG_tzTgxz1edTs?alt=media&key=AIzaSyByxcJ3WXPaoh9qg4DoCGxrbxC5LyTQvug -d checkpoints
```

then adjust the path to the test dataset in [validation.py](validation.py) and run

```bash
python3 validation.py
```