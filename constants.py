SEED = 2024

# Path variables
PATH = 'input'
SAVE_PATH = 'plots'
TRAIN_DF_PATH = 'input\stage_2_train_labels.csv'
DETAIL_DF_PATH = 'input\stage_2_detailed_class_info.csv'
TRAIN_IMAGES_PATH = 'input\stage_2_train_images'
TEST_IMAGES_PATH = 'input\stage_2_test_images'
TRAINED_MODEL_PATH = 'checkpoints/best_nn.h5'
SUBMISSION_DF_PATH = 'input\stage_2_sample_submission.csv'

# Image variables
IMAGE_SIZE = 1024
ADJUSTED_IMAGE_SIZE = 128

# Model variables
EPOCH_COUNT = 20
BATCH_SIZE = 16
VAL_BATCH_SIZE = 32
TRAIN_SAMPLES = 4800 #the same reason, we have too many images, so i decide train for sample.
VALID_SAMPLES = 1200
LEARNING_RATE = 0.0001