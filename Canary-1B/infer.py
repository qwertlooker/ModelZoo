from torch_npu.contrib import transfer_to_npu
# Load Canary model
from nemo.collections.asr.models import EncDecMultiTaskModel
import os
os.system("ln -s NeMo/nvidia/models--nvidia--canary-1b ~/.cache/huggingface/hub/models--nvidia--canary-1b")
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
  
# Transcribe
transcript = canary_model.transcribe(audio=["tianjingcheng/NeMo/nvidia/2902-9008-0000_01.wav"])
print(transcript)
# By default, Canary assumes that input audio is in English and transcribes it.
 
# To transcribe in a different language, such as Spanish
transcript = canary_model.transcribe(
     audio=["tianjingcheng/NeMo/nvidia/2902-9008-0000_01.wav"],
     batch_size=1,
     task='asr',
     source_lang='en',  # es: Spanish, fr: French, de: German
     target_lang='en',  # should be same as "source_lang" for 'asr'
     pnc='yes' )
print(transcript)

# To translate using Canary. For example, from English audio to French text
transcript = canary_model.transcribe(
     audio=["tianjingcheng/NeMo/nvidia/2902-9008-0000_01.wav"],
     batch_size=1,
     task='ast',
     source_lang='en',
     target_lang='de',  
     pnc='yes' )
print(transcript)
