import torch
from data_loader import ListDataset
from data_loader import char2token
from model import make_model
from predict import validation

batch_size = 1
val_dataloader = torch.utils.data.DataLoader(ListDataset("/path/to/the/dataset", (128, 64)), batch_size=batch_size, shuffle=False, num_workers=0)
model = make_model(len(char2token)).cuda()

checkpoint = "checkpoints/model.pth"
print(f"Checkpoint: {str(checkpoint)}")
saved = torch.load(str(checkpoint))
model.load_state_dict(saved["model"])
model.cuda()
print("Calculating ...")
cer, wer = validation(val_dataloader, model, visual=False)
print(f"CER: {cer} - WER: {wer}")
print("========")