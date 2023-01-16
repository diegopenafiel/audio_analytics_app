import os

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_CONFIG = os.path.join(ROOT_DIR, 'logging.yml')

DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_DIR_AUDIO = os.path.join(DATA_DIR, 'audio')
DATA_DIR_GUITAR = os.path.join(DATA_DIR_AUDIO, 'Guitar_Only')
DATA_DIR_AUGMENTED = os.path.join(DATA_DIR_AUDIO, 'augmented')

METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
METADATA_DIR_RAW = os.path.join(METADATA_DIR, 'raw')
METADATA_DIR_PROCESSED = os.path.join(METADATA_DIR, 'processed')

METADATA_DIR_AUGMENTED = os.path.join(METADATA_DIR, 'augmented')
METADATA_DIR_AUGMENTED_RAW = os.path.join(METADATA_DIR_AUGMENTED, 'raw')
METADATA_DIR_AUGMENTED_PROCESSED = os.path.join(METADATA_DIR_AUGMENTED, 'processed')

LOG_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_DIR_TRAINING = os.path.join(LOG_DIR, 'training')

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'whisper_class_model.pth')
# SENTIMENT_MODEL_URL = "https://azadpgsfnpqatuaensta.blob.core.windows.net/training/audio/whisper_class_model.pth?se=2023-01-12T08%3A03%3A22Z&ske=2023-01-19T07%3A03%3A22Z&skoid=5f490e73-06e5-4c43-a7c3-9f15899852f5&sks=b&skt=2023-01-12T07%3A03%3A22Z&sktid=3b618463-9352-4fa4-a67c-112da2837c29&skv=2021-04-10&sp=r&spr=https&sr=b&sv=2019-10-10&sig=tN25JCMMAeDHMICsOHVtxL%2FlSd9gIkCY5ep%2Bry3xIwU%3D"

SENTIMENT_MODEL_URL = "https://adpsentimentmodel.s3.eu-central-1.amazonaws.com/whisper_class_model.pth?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCmFwLXNvdXRoLTEiSDBGAiEAqhA700SyupgVr8QXpwvN9nJxYlZKTMYLX7xcjsOp1L4CIQCZrSA%2BUn%2B7hpXH5ANS9s3iAHAGilLDi77cVs2%2Fll5ZeSrkAggqEAAaDDM0NjI3ODQxODE5NiIMOKwVE9ceVqrsaLCFKsEC37AxEwZpgfkWrxdmi8KqWRHyQiPX5FacKDAAlHqalvtkSnCcZu2UjMY2q35FZZNsa47Jp%2F7mOVxv7vywp4MmaZPQ%2BTf0hyD3iJfS%2FZLTP0aa%2FyxGnmNQ8%2F8K1Fagoe7U9WdmT2Qtzc4On81%2BFc193MJS8TG1Eu6wLH6YxY5gPA2VGRNJcQd%2F1iZHDek6bT5QnbxMUX5ZeZrMA7RGmRrk%2F6sbIW%2FaGqFs3sNfbIbARSV%2BZ3nahQ7eS9YkAnVR70mYZjxgbK%2F5hrIAp%2BaAhw7AG6WmVr6y6A894skWMS5c8%2B6%2FjMBvpEiCxrq7cGtDuRyi3uoPIwpQPZBD8m9GY7BI35Xmt4INnusb4hV2G5gfnGoelZ8ydgiPlYXOOBzRhAQ%2F%2Bmc10vfrTvib25JDQR%2FHRIbCfn%2Fyw%2BOZwy7UgEYWFrFvMIaqlJ4GOrIC9t%2Fl3TZCWKz%2BKJ0AGaMbbP%2FePqSr%2F5IFAh1x1Zyj%2BXO0%2Fyo%2BSLMBdo%2F0AEHmuo%2B4LIYXwSEc6a3Md9HtFLV8ZQi2TcnFO9jjLo8Clj%2BvuXsKHWPBIHpzNXznDcIEHG9PP7VtGKQt1BSht1aO4iIQqxMM6niL2RW4cvLMkeZi4okO07ChEntGQIuHBFWr0YjJoLnJn%2B%2B0Nhy7Ko%2B3GXGQ8ms%2BF5j4qfyXSUfGhArvZR4Ctxztz5EDSFJPlWjMJDmYFdgrvKv2dXJajUvlzM7Y5LDCpoqDnwYWWEv5Zx3B6cf2iApIl%2BuCPAS2eQtf3vZrLOWa%2BLQ%2BTbAWosEK3GD%2FIAdQxdWTliSbc1ZGE8GVIdkXm3qdTQ0Lvkcd%2FvJQ0IusabfFzxJeLTRqvi%2BL3FZG9FLX&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230116T091249Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAVBH6NK4KMSMZT5H4%2F20230116%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=bc76cb6b826dbee7ee349acc59646f35007adaa5641f775ef30b5a417c328431"

OUT_DIR = os.path.join(ROOT_DIR, 'output')
RECORDING_DIR = os.path.join(OUT_DIR, 'recording')
IMAGE_DIR = os.path.join(OUT_DIR, 'images')

WAVE_OUTPUT_FILE = os.path.join(RECORDING_DIR, "recorded.wav")
SPECTROGRAM_FILE = os.path.join(RECORDING_DIR, "spectrogram.png")

# Audio configurations
INPUT_DEVICE = 0
MAX_INPUT_CHANNELS = 1  # Max input channels
DEFAULT_SAMPLE_RATE = 16000   # Default sample rate of microphone or recording device
DURATION = 5   # 5 seconds
CHUNK_SIZE = 1024