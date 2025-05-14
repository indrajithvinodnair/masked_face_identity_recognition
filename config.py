# Base path for the LFW dataset
BASEPATH = '/root/face_recog/lfw/'

# Paths for dataset merging
# this is the path to the lfw-deepfunneled dataset inside lfw folder
ORIGINAL_LFW_PATH = BASEPATH + 'lfw-deepfunneled/lfw-deepfunneled'
# this is the path to the dataset with the masks
MASKED_LFW_PATH = BASEPATH + 'lfw-deepfunneled/lfw-deepfunneled_masked'
# this is the path of the merged dataset
OUTPUT_MERGED_PATH = BASEPATH + 'lfw_merged'

# Paths for dataset splitting
INPUT_MERGED_DATASET = BASEPATH + 'lfw_merged'
OUTPUT_SPLIT_PATH = BASEPATH + 'lfw_train_test_eval'

# Paths for training and evaluation
DATASET_ROOT = BASEPATH + 'lfw_train_test_eval'
TEST_ROOT = BASEPATH + 'lfw_train_test_eval/test'

# This is where the models and results will be stored to 
BASELINE_RESULTS_PATH = BASEPATH.replace('/lfw/', '/')+"baseline_results"

