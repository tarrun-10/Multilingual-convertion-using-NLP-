```pip install pytesseract # For recognising and extracting the text from the image

!sudo apt install tesseract-ocr #OCR developed by google it can sence 100 languages

import pytesseract
import shutil #Provides high level interfacing 
import os  # provides functions for interacting with the operating system 
import random # Generates random numbers 
import cv2 # For computer vision functionalities for image and video processing.
import numpy as np #provides support for working with arrays and matrices of numerical data
import matplotlib.pyplot as plt # used for creating visualizations and plots
from PIL import Image

ori_image = cv2.imread("/content/hqdefault.jpg") #Extracting the image 
ori_img = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) # Appyling color
plt.imshow(ori_img) 
plt.axis('off')
plt.show()

fixed_img = cv2.resize(ori_img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
plt.imshow(fixed_img)
plt.axis("off")

ogimg = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY) #to convert an RGB image to gray scale 

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) # library to create a structuring element for morphological operations.
dilation = cv2.dilate(ogimg, kernel, iterations=1) #applying dilation 
plt.imshow(dilation)
plt.axis("off")

binary = cv2.threshold(cv2.medianBlur(erosion, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # thresholding for converting the image to bianry
plt.imshow(binary)
plt.axis("off")

gsbin = binary.astype(np.uint8) #storing the binary image in 8 bit
# gsbin= 255 * gsbin
plt.imshow(gsbin)

image2 = cv2.cvtColor(255-gsbin, cv2.COLOR_GRAY2RGB) #to convert a grayscale image represented by the 255-gsbin variable to an RGB image.
plt.imshow(image2)
plt.axis('off')
plt.show()

import matplotlib.image
matplotlib.image.imsave('image1.png', image2)

extractedText = pytesseract.image_to_string("image1.png",lang='eng+tam+hin+kan+tel+mal',config='--psm 6') #convert the extracted to string
extractedText= extractedText.replace('\n', ' ')

print(extractedText)

!pip install langdetect #detect the language 

from langdetect import detect
lang = detect(extractedText)
print(lang)

pip install translate #to translate text from one language to another

pip install gTTs #Google Text-to-Speech

pip install gtts

import os
import gtts as gt

from translate import Translator
translator= Translator(from_lang=lang,to_lang="en")
translation = translator.translate(extractedText)
print(translation)

trans = ""
i=0
preele= ""
for element in translation:
    i+=1
    n = ord(element)
    if 97 <= n <= 122 or 65<=n<=90 or 48<=n<=57:
        trans=trans+element
    else:
        if(preele==" "):
            trans=trans+""
        else:
            trans=trans+" "
    preele=element
      #print(element, end=' ')
    if(i==499):
        break;
print(trans)

trans1 = ""
i=0
preele= ""
for element in trans:
    i+=1
    n = ord(element)
    if 97 <= n <= 122 or 65<=n<=90 or 48<=n<=57:
        trans1=trans1+element
    else:
        if(preele==" "):
            trans1=trans1+""
        else:
            trans1=trans1+" "
    preele=element
      #print(element, end=' ')
    if(i==499):
        break;
print(trans1)

from textblob import TextBlob
tb_txt = TextBlob(trans1)

correctedTBText = tb_txt.correct()
correctedText = str(correctedTBText)

print(str(correctedText)) #finale text after modifications 

!pip install git+https://github.com/huggingface/transformers -q  #importing the Dataset from hugging face 
!pip install transformers -U -q  #Providies pretrained models and tokenzing tools 
!pip install sentencepiece #open-source text tokenizer and detokenizer developed by Google
!pip freeze | grep transformers #filters the output 

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast # they are pretrained models used for convertion one language to another 

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt") #downloading the pretrained model 
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

model_inputs = tokenizer(correctedText, return_tensors="pt") #tokenizing the input 

generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"] #setting the language 
)

translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(translation)

!pip install playsound #library to play sound 

!pip install pydub #library for dubbing 

pip install ffmpeg #library for converting to speech 

txt=' '.join(translation) #convert the list to string
tts=gt.gTTS(text=txt,lang="ta") 
tts.save("ttso.wav") #speech generated 
os.system("ttso.wav")

from IPython.display import Audio, display
sound_file = 'ttso.wav'
display(Audio(sound_file, autoplay=True)) #output 

```

