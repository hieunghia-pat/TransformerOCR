## dataset configuartion
batch_size = 16
out_level = "character"
image_dirs = ["../UIT-HWDB-dataset/UIT_HWDB_line/train_data"] # for making vocab
train_image_dirs = ["../UIT-HWDB-dataset/UIT_HWDB_line/train_data"] # for training
test_image_dirs = ["../UIT-HWDB-dataset/UIT_HWDB_line/test_data"] # for testing
image_size = (1024, 64)

## training configuration
max_epoch = 100
learning_rate = 5e-4
checkpoint_path = "/content/gdrive/MyDrive/TransformerOCR/saved_models"
saved_optimizer = None
start_from = None
warmup = 2000
factor = 1

## model configuration
dropout = 0.5
num_layers = 4
d_model = 256
dff = 512
heads = 8
beam_size = 2

## objective function configuration
smoothing = 0.2

## configure for debug only
debug = True
save_per_iter = 100