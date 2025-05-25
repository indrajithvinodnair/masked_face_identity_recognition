# Base path for the LFW dataset
BASEPATH = '/root/face_recog/lfw/'

# Paths for dataset merging
# this is the path to the lfw-deepfunneled dataset inside lfw folder
UNMASKED_LFW_PATH = BASEPATH + 'lfw-deepfunneled/lfw-deepfunneled'
# this is the path to the dataset with the masks
MASKED_LFW_PATH = BASEPATH + 'lfw-deepfunneled/lfw-deepfunneled_masked'
# this is the path of the merged dataset
MASKED_AND_UNMASKED_MERGED_PATH = BASEPATH + 'lfw_merged'

# Paths for dataset splitting
UNMASKED_DATASET_SPLIT_PATH = BASEPATH + 'lfw_train_test_eval/unmasked'
MASKED_DATASET_SPLIT_PATH = BASEPATH + 'lfw_train_test_eval/masked'



# Paths for training and evaluation
DATASET_ROOT = BASEPATH + 'lfw_train_test_eval'
TEST_ROOT = BASEPATH + 'lfw_train_test_eval/test'

# This is where the models and results will be stored to 
BASELINE_RESULTS_PATH = BASEPATH.replace('/lfw/', '/')+"baseline_results"

# Mask keywords to identify masked images
mask_keywords = ["n95", "surgical", "kn95"]

