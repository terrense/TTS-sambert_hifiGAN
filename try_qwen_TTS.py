# Please install the latest version of the DashScope SDK.
import os
import requests
import dashscope

text = "Let me recommend a T-shirt to everyone. This one is really super good-looking, and the color is very classy. It’s also a great piece to mix and match with anything, so you can totally buy it without hesitation. It looks amazing and is very forgiving on the figure—no matter what body type you have, it will look great on you. Highly recommend you place an order!"
text = "您好，我是由华工科技公司和同济医院联合开发的医疗智能体小护，请问有什么可以帮助您的呢？"
# Usage of the SpeechSynthesizer Interface: dashscope.audio.qwen_tts.SpeechSynthesizer.call(...)
# response = dashscope.MultiModalConversation.call(
#     model="qwen3-tts-flash-2025-11-27",
#     api_key="sk-60f3014ee69d42d4956c37c88cf3085a",
#     text=text,
#     voice="Ryan",
#     language_type="English", # It is recommended to match the language with the text in order to obtain correct pronunciation and natural intonation.
#     stream=False
# )

response = dashscope.MultiModalConversation.call(
    model="qwen3-tts-flash-2025-11-27",
    api_key="sk-60f3014ee69d42d4956c37c88cf3085a",
    text=text,
    voice="Bunny",
    language_type="Chinese", # It is recommended to match the language with the text in order to obtain correct pronunciation and natural intonation.
    stream=False
)

audio_url = response.output.audio.url
save_path = "downloaded_audio.wav"  # Custom save path.

try:
    response = requests.get(audio_url)
    response.raise_for_status()  # Check whether the request was successful.
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"The audio file has been saved to: {save_path}")
except Exception as e:
    print(f"Download failed: {str(e)}")







# =================================================================================================
# coding=utf-8

import dashscope
from dashscope.audio.tts_v2 import *

# If you have not configured the API Key in an environment variable, replace "your-api-key" with your API Key.
dashscope.api_key = "sk-60f3014ee69d42d4956c37c88cf3085a"

# Model
# Different model versions require corresponding voice versions:
# cosyvoice-v3-flash/cosyvoice-v3-plus: Use voices such as longanyang.
# cosyvoice-v2: Use voices such as longxiaochun_v2.
model = "cosyvoice-v3-flash"
# Voice
voice = "longanyang"

# Instantiate SpeechSynthesizer and pass request parameters such as model and voice in the constructor.
synthesizer = SpeechSynthesizer(model=model, voice=voice)
# Send the text to be synthesized to get the binary audio.
audio = synthesizer.call("您好，我是由华工科技公司和同济医院联合开发的医疗智能体小护，请问有什么可以帮助您的呢？")
# The first time you send text, a WebSocket connection must be established, so the first-packet latency includes the connection setup time.
print('[Metric] Request ID: {}, First-packet latency: {} ms'.format(
    synthesizer.get_last_request_id(),
    synthesizer.get_first_package_delay()))

# Save the audio to a local file.
with open('output.mp3', 'wb') as f:
    f.write(audio)