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

SENTIMENT_MODEL_URL = "https://azadpgsfnpqatuaensta.blob.core.windows.net/training/audio/whisper_class_model.pth?se=2023-01-12T08%3A03%3A22Z&ske=2023-01-19T07%3A03%3A22Z&skoid=5f490e73-06e5-4c43-a7c3-9f15899852f5&sks=b&skt=2023-01-12T07%3A03%3A22Z&sktid=3b618463-9352-4fa4-a67c-112da2837c29&skv=2021-04-10&sp=r&spr=https&sr=b&sv=2019-10-10&sig=tN25JCMMAeDHMICsOHVtxL%2FlSd9gIkCY5ep%2Bry3xIwU%3D"

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