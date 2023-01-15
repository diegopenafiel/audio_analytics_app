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
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')
# SENTIMENT_MODEL_URL = "https://azadpgsfnpqatuaensta.blob.core.windows.net/training/audio/whisper_class_model.pth?se=2023-01-12T08%3A03%3A22Z&ske=2023-01-19T07%3A03%3A22Z&skoid=5f490e73-06e5-4c43-a7c3-9f15899852f5&sks=b&skt=2023-01-12T07%3A03%3A22Z&sktid=3b618463-9352-4fa4-a67c-112da2837c29&skv=2021-04-10&sp=r&spr=https&sr=b&sv=2019-10-10&sig=tN25JCMMAeDHMICsOHVtxL%2FlSd9gIkCY5ep%2Bry3xIwU%3D"

SENTIMENT_MODEL_URL = "https://adpsentimentmodel.s3.eu-central-1.amazonaws.com/whisper_class_model.pth?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCWV1LXdlc3QtMyJHMEUCIQDu7ed1RjwVQzBxfgeXWjWpeeV77zZFjpVV8gldRgr0HAIgGynuXYUfP%2FhOARXZ3ITND7JHk0V%2FbkJqH8UxQX5jhnIq5AIIEBAAGgwzNDYyNzg0MTgxOTYiDFmriL8D1jEP0O02HirBAgA1yPJcC2L73suOoBk7TC%2FdufT%2Fz18YqDKAJu1uMkW8D4kojHwfhN4D8Hv0t5qLMdkxezl7Qv%2FuXqPKmLGVgVgcS37PVmWyklSmLZvbUL6g%2BlnEZabw1VhMU9q4wB0ot7M6Dn0v33fMO3vV9R%2B%2F7gYc4V48xTe9wePJX7KlnHfnAQ7Sk6APYgegxh4OfnAO5eSA9BBAs4yuueE3HSH5TlLbwDIGQgJwvo4VRsJka56NKSX%2Fq264Nnrc8LztLiPuNGFoSV1p9QAbcTZ1J7f0sNg1FudM%2BsFqkTWAFYQ4GhWYu5%2FiPEcucm8uodKoy6pKrmzRPzEOIZkYiby6nOpYP%2F0ePVkJVFD%2BWL4PqbfaCXlvBxQx4QOZB%2FciH2vgmLLwYdxkN%2BtXJUvm%2B8jRbI%2BH56TEpA17WMlh%2FdhsusRkI8zEzDDN0I6eBjqzAqnpJDrTJcXlH2k6mqnL1x3awD849ZgSvNUrdM1F5%2FE7UF3LdI7bId2O9yCUyRQr31md1D5KI0TLax3LbUL5xcGF7xrjXxRN0NflZjg2ifRInITDq%2FNwlLjV5DuDVc9o7qCxIy4EmGMY0GqDrk66kfq8i6WEFryRnKo1LxL4Goj92z6QZBN8cvAxo1xIZD7Ej%2ByrzfiwJnrJKoqtOsVdicu1mc0iRqZNJ794VrJU8KCERTWoZ7E7pePcR2DTQKomdhyqWiERTkn2%2BWpRUSHGeCVuaAOxWy9ZzZchpUixg2%2FXr6ZO0hRYAGWEVIL0qp6i5S4DGptILu90ilcPLgXrqZnfOqH7C3SYPPRcL%2FojM2EtHcrQicNas0Sn8AYeVqwnJux7UGIwABJpPl4Qvv76Oa2FN7A%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230115T072230Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAVBH6NK4KEPGFQ7PZ%2F20230115%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=9002f6a19f79208e841b7219796fed0b8a8a14902f0e7d9fb3529dcf43efe6af"

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