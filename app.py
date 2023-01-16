import sys, os

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
import numpy as np
import time 
import json
import logging
from settings import (
    DURATION, DEFAULT_SAMPLE_RATE, MAX_INPUT_CHANNELS,
    WAVE_OUTPUT_FILE, INPUT_DEVICE, CHUNK_SIZE,
    RECORDING_DIR, IMAGE_DIR,
    SENTIMENT_MODEL_URL, MODEL_PATH, MODEL_DIR
)
import queue
import urllib.request
from pathlib import Path

import math
import librosa
import torch
import whisper
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor, 
    WhisperModel
)
from typing import Optional, Tuple, Union

SENT_CLASSES = [
      'neutral', 
      'calm', 
      'happy', 
      'sad', 
      'angry', 
      'fearful', 
      'disgust', 
      'surprised'
    ]

class WhisperClassificationHead(torch.nn.Module):
    """Head for classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, inner_dim)
        self.dropout = torch.nn.Dropout(p=pooler_dropout)
        self.out_proj = torch.nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class WhisperForSentimentAnalysis(torch.nn.Module):
    def __init__(
        self, 
        whisper_model: str,
        num_classes: int,
        classifier_dropout: float,
        ):
        super().__init__()

        self.model = WhisperModel(WhisperConfig.from_pretrained(whisper_model))

        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.classification_head = WhisperClassificationHead(
            self.model.config.d_model,
            self.model.config.d_model,
            self.num_classes,
            self.classifier_dropout,
        )

    def forward(self,
        input_features: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_features,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        speech_representation = hidden_states[:, -1, :]
        logits = self.classification_head(speech_representation)
        return logits

def get_split_audio(processor, file_path, resampling_rate=16000, split_length=15):
    # retrieve audio file and load it
    data, sr = librosa.load(file_path)
    
    #get audio file in numpy array and sr as original sampling rate
    buffer = split_length * sr
    samples_total = len(data)
    samples_wrote = 0
    counter = 1

    while samples_wrote < samples_total:
        #check if the buffer is not exceeding total samples 
        if buffer > (samples_total - samples_wrote):
            buffer = samples_total - samples_wrote
        block = data[samples_wrote : (samples_wrote + buffer)]
        if resampling_rate is not None:
            block = librosa.resample(block, orig_sr=sr, target_sr=resampling_rate)

        #create input split_length second segment
        input_features = processor(block, return_tensors="pt", sampling_rate=resampling_rate).input_features
        if counter == 1:
            all_input_features = input_features
        else:
            all_input_features = torch.cat((all_input_features, input_features), 0)
        counter += 1
        samples_wrote += buffer
    
    return all_input_features

def run_sentiment(model, all_input_features, batch_size = 8):
    all_probabilities = []

    for i in range(math.ceil(all_input_features.shape[0]/batch_size)):
        # Generate logits
        decoder_input_ids = torch.tensor(all_input_features[i*batch_size:(i+1)*batch_size,:,:].shape[0] * [[1]]) * model.model.config.eos_token_id
        logits = model(all_input_features[i*batch_size:(i+1)*batch_size,:,:], decoder_input_ids=decoder_input_ids)

        # take softmax
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)
        probs = probs.detach().cpu().numpy()
        all_probabilities.extend(probs)
    
    return all_probabilities

# Transcribe text
# @st.cache(persist=True, max_entries=3, ttl=300)
# @st.experimental_singleton 
def transcribe(file_path, model_option):
    if not os.path.exists(file_path):
        raise Exception("Audio path does not exists.")
    # load transcription model
    model = whisper.load_model(model_option)
    # generate transcript
    transcript = model.transcribe(file_path)

    return transcript

# Sentiment Analysis
def sentiment_analysis(file_path, sentiment_model_path, emotion_labels, threshold=0.8, batch_size=8):
    # load model and processor
    whisper_model = "openai/whisper-tiny.en"
    
    num_classes = len(emotion_labels)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
    model = WhisperForSentimentAnalysis(whisper_model=whisper_model, 
                                        num_classes=num_classes, 
                                        classifier_dropout=0.1)
    state_dict = torch.load(sentiment_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    all_input_features = get_split_audio(feature_extractor, file_path, resampling_rate=16000, split_length=15)
    all_probabilities = run_sentiment(model, all_input_features, batch_size=8)

    return all_probabilities

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

def sst(path_to_wav):
    def recorder_factory():
        return MediaRecorder(path_to_wav)
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        in_recorder_factory=recorder_factory,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )
    
    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return
    
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

def main():
    #Main Body    
    head1, head2 = st.columns(2)
    
    with head1:
        st.image(os.path.join(IMAGE_DIR, "ADP.png"), use_column_width='auto')
    with head2:
        st.image(os.path.join(IMAGE_DIR, "MBZUAI.png"), use_column_width='auto')
    
    st.header('Call Centre Audio Analytics')
    st.write('In this application we leverage deep learning models to process and analyse human speech.')
    
    download_file(SENTIMENT_MODEL_URL, Path(MODEL_PATH), expected_size=151710983)
    
    # audio recording
    sst(path_to_wav=WAVE_OUTPUT_FILE)
    if os.path.exists(WAVE_OUTPUT_FILE):
        st.audio(WAVE_OUTPUT_FILE)

    # Tasks
    col1, col2 = st.columns(2)

    with col1:
        gen_models = [
                "tiny",
                # "base",
                # "small",
                # "medium",
                # "large",
            ]
        model_option = st.selectbox(
            'Select a transcription engine:',
            tuple(gen_models),
        )
        model_option = str(model_option)
        if st.button('Run Transcript Generation'): 
            try:
                with st.spinner(f'Generating transcript...'):
                    transcript = transcribe(WAVE_OUTPUT_FILE, model_option=model_option)
                st.subheader('Audio transcript:')
                for s in transcript['segments']:
                    st.write(s['text'])
            except:
                st.error("No audio has been recorded.")

    with col2:
        threshold = st.slider('Select a confidence threshold:', 0.0, 100.0, 50.0)

        if st.button('Run Sentiment Analysis'):
            try:
                with st.spinner(f'Analysing audio...'):
                    all_probabilities = sentiment_analysis(file_path=WAVE_OUTPUT_FILE, 
                                                           sentiment_model_path=MODEL_PATH, 
                                                           emotion_labels=SENT_CLASSES,
                                                           threshold=float(threshold) / 100.0)
                st.subheader("Sentiment(s) found:")
                for i in range(len(all_probabilities)):
                    emotion_idx = np.argmax(all_probabilities[i])
                    emotion_pred = SENT_CLASSES[emotion_idx]
                    if all_probabilities[i][emotion_idx] > (threshold/100):
                        st.write("Emotion detected:", str(emotion_pred), 
                              "with probability", "%d %%" % (all_probabilities[i][emotion_idx] * 100))
            except Exception as e:
                st.error(e)
                # st.error("No audio has been recorded.")

if __name__ == '__main__':
    main()
