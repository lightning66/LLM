#!/usr/bin/env python
# coding: utf-8

# ### Definition:

# In this application, Turkish and Farsi languages are converted to each other. It is a 2 way translation. <br>
# The goal is to perform translation in the shortest way possible.

# ### Structure:

# Speak a Language   -->   Transcribe it   -->    Convert to the other language    -->   Text to speech conversion

# In[3]:


#Import
# imports
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


# In[4]:


# Initialization
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()


# In[18]:


def transcribe_audio(audio_path, language):
    if not audio_path or not os.path.exists(audio_path):
        return "No audio file found."
        
    client = OpenAI()
    audio_file = open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        language=language,
    )
    return transcription.text

def chat2turkish(message, history):
    system_message_tr = "Translate the message to Turkish"
    messages = [{"role": "system", "content": system_message_tr}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

def chat2farsi(message, history):
    system_message_fa = "Translate the message to Farsi"
    messages = [{"role": "system", "content": system_message_fa}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def text_translate(message, language):
    history = []
    if language == 'fa':
        translated_txt = chat2turkish(message, history)
    elif language == 'tr':
        translated_txt = chat2farsi(message, history)

    return translated_txt


def text2speech(message):
    response = openai.audio.speech.create(
      model="tts-1",
      voice="onyx",    # Also, try replacing onyx with alloy
      input=message
    )
    
    path = "translation.mp3"
    # Save the response content as an MP3 file
    audio_file_path = path
    with open(audio_file_path, "wb") as f:
        f.write(response.content)

    return path


# In[19]:


from pathlib import Path

def voice_to_voice(audio_path, source_language):
    # Transcribe the audio
    transcription_response = transcribe_audio(audio_path, source_language)
    # Translate 
    translated_text = text_translate(transcription_response, source_language)
    # Convert texts to speech
    translated_audio_path = text2speech(translated_text)
    audio_path = Path(translated_audio_path)
    return audio_path, transcription_response, translated_text


# In[20]:


audio_input = gr.Audio(
    sources=["microphone"],
    type="filepath"
)

demo = gr.Interface(
    fn = voice_to_voice,
    inputs = [audio_input, gr.Dropdown(["fa", "tr"], label="Source Language", value="fa")],
    outputs = [gr.Audio(label="Play Translation"), gr.Textbox(label="Original Text:", lines=3),
               gr.Textbox(label="Translated Text:", lines=3)]
)

demo.launch(server_name="0,0,0,0")


# In[ ]:




