```pip install pytesseract # For recognising and extracting the text from the image ```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting pytesseract
  Downloading pytesseract-0.3.10-py3-none-any.whl (14 kB)
Requirement already satisfied: Pillow>=8.0.0 in /usr/local/lib/python3.9/dist-packages (from pytesseract) (8.4.0)
Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.9/dist-packages (from pytesseract) (23.0)
Installing collected packages: pytesseract
Successfully installed pytesseract-0.3.10
```!sudo apt install tesseract-ocr #OCR developed by google it can sence 100 languages```
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following additional packages will be installed:
  tesseract-ocr-eng tesseract-ocr-osd
The following NEW packages will be installed:
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd
0 upgraded, 3 newly installed, 0 to remove and 24 not upgraded.
Need to get 4,850 kB of archives.
After this operation, 16.3 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu focal/universe amd64 tesseract-ocr-eng all 1:4.00~git30-7274cfa-1 [1,598 kB]
Get:2 http://archive.ubuntu.com/ubuntu focal/universe amd64 tesseract-ocr-osd all 1:4.00~git30-7274cfa-1 [2,990 kB]
Get:3 http://archive.ubuntu.com/ubuntu focal/universe amd64 tesseract-ocr amd64 4.1.1-2build2 [262 kB]
Fetched 4,850 kB in 2s (1,981 kB/s)
debconf: unable to initialize frontend: Dialog
debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 3.)
debconf: falling back to frontend: Readline
debconf: unable to initialize frontend: Readline
debconf: (This frontend requires a controlling tty.)
debconf: falling back to frontend: Teletype
dpkg-preconfigure: unable to re-open stdin: 
Selecting previously unselected package tesseract-ocr-eng.
(Reading database ... 122349 files and directories currently installed.)
Preparing to unpack .../tesseract-ocr-eng_1%3a4.00~git30-7274cfa-1_all.deb ...
Unpacking tesseract-ocr-eng (1:4.00~git30-7274cfa-1) ...
Selecting previously unselected package tesseract-ocr-osd.
Preparing to unpack .../tesseract-ocr-osd_1%3a4.00~git30-7274cfa-1_all.deb ...
Unpacking tesseract-ocr-osd (1:4.00~git30-7274cfa-1) ...
Selecting previously unselected package tesseract-ocr.
Preparing to unpack .../tesseract-ocr_4.1.1-2build2_amd64.deb ...
Unpacking tesseract-ocr (4.1.1-2build2) ...
Setting up tesseract-ocr-eng (1:4.00~git30-7274cfa-1) ...
Setting up tesseract-ocr-osd (1:4.00~git30-7274cfa-1) ...
Setting up tesseract-ocr (4.1.1-2build2) ...
Processing triggers for man-db (2.9.1-1) ...
```
import pytesseract
import shutil #Provides high level interfacing 
import os  # provides functions for interacting with the operating system 
import random # Generates random numbers 
import cv2 # For computer vision functionalities for image and video processing.
import numpy as np #provides support for working with arrays and matrices of numerical data
import matplotlib.pyplot as plt # used for creating visualizations and plots
from PIL import Image
```
### **Extracting the text from the image**
```
ori_image = cv2.imread("/content/hqdefault.jpg") #Extracting the image 
ori_img = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) # Appyling color
plt.imshow(ori_img) 
plt.axis('off')
plt.show()
```
![image](https://user-images.githubusercontent.com/110376310/232060004-f5ab5370-6cb7-478a-be23-f27242fdb47b.png)
```
fixed_img = cv2.resize(ori_img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
plt.imshow(fixed_img)
plt.axis("off")
```

![image](https://user-images.githubusercontent.com/110376310/232060072-fcf788b3-cb0b-42e1-b953-119598ceb9e0.png)

```
ogimg = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY) #to convert an RGB image to gray scale 
```
```
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) # library to create a structuring element for morphological operations.
dilation = cv2.dilate(ogimg, kernel, iterations=1) #applying dilation 
plt.imshow(dilation)
plt.axis("off")
```
![image](https://user-images.githubusercontent.com/110376310/232060165-4f924598-d5f4-402c-81bc-a74911a973c0.png)
```
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
erosion= cv2.erode(dilation, kernel, iterations=1) # erostion 
plt.imshow(erosion)
plt.axis("off")
```
![image](https://user-images.githubusercontent.com/110376310/232060549-45e0633c-8c0a-4892-9a05-0386a12c645f.png)

```
binary = cv2.threshold(cv2.medianBlur(erosion, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # thresholding for converting the image to bianry
plt.imshow(binary)
plt.axis("off")
```
![image](https://user-images.githubusercontent.com/110376310/232060648-d54016de-a1d2-4507-a46a-c48ba313c49c.png)


```
gsbin = binary.astype(np.uint8) #storing the binary image in 8 bit
# gsbin= 255 * gsbin
plt.imshow(gsbin)
```
![image](https://user-images.githubusercontent.com/110376310/232060685-b9aa6e4d-4e88-41e1-ad8e-27fa8aa2964b.png)
```

image2 = cv2.cvtColor(255-gsbin, cv2.COLOR_GRAY2RGB) #to convert a grayscale image represented by the 255-gsbin variable to an RGB image.
plt.imshow(image2)
plt.axis('off')
plt.show()
```
![image](https://user-images.githubusercontent.com/110376310/232060736-53daae57-a82f-420e-928c-97f61b3f10ec.png)
```
import matplotlib.image
matplotlib.image.imsave('image1.png', image2)
```
```
extractedText = pytesseract.image_to_string("image1.png",lang='eng+tam+hin+kan+tel+mal',config='--psm 6') #convert the extracted to string
extractedText= extractedText.replace('\n', ' ')
```
```
print(extractedText)
```
LINGUANAUT I'm learning Spanish right now. because | think it's a beautiful language. and also because | want to visit Spain one day. I'm improving day after day. but | need to practice with someone who Is a native. Siw MUIR CTU meroli) 

### **Language detection and making the text more understandable**
```
!pip install langdetect #detect the language 
```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting langdetect
  Downloading langdetect-1.0.9.tar.gz (981 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 981.5/981.5 kB 48.5 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Requirement already satisfied: six in /usr/local/lib/python3.9/dist-packages (from langdetect) (1.16.0)
Building wheels for collected packages: langdetect
  Building wheel for langdetect (setup.py) ... done
  Created wheel for langdetect: filename=langdetect-1.0.9-py3-none-any.whl size=993243 sha256=6317567ae726b42b3aff98f90445cbb2bea0f08c905cc910c385290b35a4a815
  Stored in directory: /root/.cache/pip/wheels/d1/c1/d9/7e068de779d863bc8f8fc9467d85e25cfe47fa5051fff1a1bb
Successfully built langdetect
Installing collected packages: langdetect
Successfully installed langdetect-1.0.9
```
from langdetect import detect
lang = detect(extractedText)
print(lang)
```
en
```
pip install translate #to translate text from one language to another
```
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting translate
  Downloading translate-3.6.1-py2.py3-none-any.whl (12 kB)
Requirement already satisfied: lxml in /usr/local/lib/python3.9/dist-packages (from translate) (4.9.2)
Collecting libretranslatepy==2.1.1
  Downloading libretranslatepy-2.1.1-py3-none-any.whl (3.2 kB)
Requirement already satisfied: click in /usr/local/lib/python3.9/dist-packages (from translate) (8.1.3)
Requirement already satisfied: requests in /usr/local/lib/python3.9/dist-packages (from translate) (2.27.1)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests->translate) (3.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests->translate) (2022.12.7)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests->translate) (2.0.12)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests->translate) (1.26.15)
Installing collected packages: libretranslatepy, translate
Successfully installed libretranslatepy-2.1.1 translate-3.6.1

```
pip install gTTs #Google Text-to-Speech
```

Downloading the pakages for Image Extraction and Text detection
[ ]
pip install pytesseract # For recognising and extracting the text from the image 
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting pytesseract
  Downloading pytesseract-0.3.10-py3-none-any.whl (14 kB)
Requirement already satisfied: Pillow>=8.0.0 in /usr/local/lib/python3.9/dist-packages (from pytesseract) (8.4.0)
Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.9/dist-packages (from pytesseract) (23.0)
Installing collected packages: pytesseract
Successfully installed pytesseract-0.3.10
[ ]
!sudo apt install tesseract-ocr #OCR developed by google it can sence 100 languages
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following additional packages will be installed:
  tesseract-ocr-eng tesseract-ocr-osd
The following NEW packages will be installed:
  tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd
0 upgraded, 3 newly installed, 0 to remove and 24 not upgraded.
Need to get 4,850 kB of archives.
After this operation, 16.3 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu focal/universe amd64 tesseract-ocr-eng all 1:4.00~git30-7274cfa-1 [1,598 kB]
Get:2 http://archive.ubuntu.com/ubuntu focal/universe amd64 tesseract-ocr-osd all 1:4.00~git30-7274cfa-1 [2,990 kB]
Get:3 http://archive.ubuntu.com/ubuntu focal/universe amd64 tesseract-ocr amd64 4.1.1-2build2 [262 kB]
Fetched 4,850 kB in 2s (1,981 kB/s)
debconf: unable to initialize frontend: Dialog
debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 3.)
debconf: falling back to frontend: Readline
debconf: unable to initialize frontend: Readline
debconf: (This frontend requires a controlling tty.)
debconf: falling back to frontend: Teletype
dpkg-preconfigure: unable to re-open stdin: 
Selecting previously unselected package tesseract-ocr-eng.
(Reading database ... 122349 files and directories currently installed.)
Preparing to unpack .../tesseract-ocr-eng_1%3a4.00~git30-7274cfa-1_all.deb ...
Unpacking tesseract-ocr-eng (1:4.00~git30-7274cfa-1) ...
Selecting previously unselected package tesseract-ocr-osd.
Preparing to unpack .../tesseract-ocr-osd_1%3a4.00~git30-7274cfa-1_all.deb ...
Unpacking tesseract-ocr-osd (1:4.00~git30-7274cfa-1) ...
Selecting previously unselected package tesseract-ocr.
Preparing to unpack .../tesseract-ocr_4.1.1-2build2_amd64.deb ...
Unpacking tesseract-ocr (4.1.1-2build2) ...
Setting up tesseract-ocr-eng (1:4.00~git30-7274cfa-1) ...
Setting up tesseract-ocr-osd (1:4.00~git30-7274cfa-1) ...
Setting up tesseract-ocr (4.1.1-2build2) ...
Processing triggers for man-db (2.9.1-1) ...
[ ]
# Importing the basic pakages required for the code 
import pytesseract
import shutil #Provides high level interfacing 
import os  # provides functions for interacting with the operating system 
import random # Generates random numbers 
import cv2 # For computer vision functionalities for image and video processing.
import numpy as np #provides support for working with arrays and matrices of numerical data
import matplotlib.pyplot as plt # used for creating visualizations and plots
from PIL import Image
Extracting the text from the image
[ ]
# original image
ori_image = cv2.imread("/content/hqdefault.jpg") #Extracting the image 
ori_img = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) # Appyling color
plt.imshow(ori_img) 
plt.axis('off')
plt.show()

[ ]
fixed_img = cv2.resize(ori_img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
plt.imshow(fixed_img)
plt.axis("off")

[ ]
ogimg = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY) #to convert an RGB image to gray scale 
[ ]
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) # library to create a structuring element for morphological operations.
dilation = cv2.dilate(ogimg, kernel, iterations=1) #applying dilation 
plt.imshow(dilation)
plt.axis("off")

[ ]
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
erosion= cv2.erode(dilation, kernel, iterations=1) # erostion 
plt.imshow(erosion)
plt.axis("off")

[ ]
binary = cv2.threshold(cv2.medianBlur(erosion, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # thresholding for converting the image to bianry
plt.imshow(binary)
plt.axis("off")

[ ]
gsbin = binary.astype(np.uint8) #storing the binary image in 8 bit
# gsbin= 255 * gsbin
plt.imshow(gsbin)

[ ]
image2 = cv2.cvtColor(255-gsbin, cv2.COLOR_GRAY2RGB) #to convert a grayscale image represented by the 255-gsbin variable to an RGB image.
plt.imshow(image2)
plt.axis('off')
plt.show()

[ ]
import matplotlib.image
matplotlib.image.imsave('image1.png', image2)
[ ]
extractedText = pytesseract.image_to_string("image1.png",lang='eng+tam+hin+kan+tel+mal',config='--psm 6') #convert the extracted to string
extractedText= extractedText.replace('\n', ' ')
[ ]
print(extractedText)
LINGUANAUT I'm learning Spanish right now. because | think it's a beautiful language. and also because | want to visit Spain one day. I'm improving day after day. but | need to practice with someone who Is a native. Siw MUIR CTU meroli) 
Language detection and making the text more understandable
[ ]
!pip install langdetect #detect the language 
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting langdetect
  Downloading langdetect-1.0.9.tar.gz (981 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 981.5/981.5 kB 48.5 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Requirement already satisfied: six in /usr/local/lib/python3.9/dist-packages (from langdetect) (1.16.0)
Building wheels for collected packages: langdetect
  Building wheel for langdetect (setup.py) ... done
  Created wheel for langdetect: filename=langdetect-1.0.9-py3-none-any.whl size=993243 sha256=6317567ae726b42b3aff98f90445cbb2bea0f08c905cc910c385290b35a4a815
  Stored in directory: /root/.cache/pip/wheels/d1/c1/d9/7e068de779d863bc8f8fc9467d85e25cfe47fa5051fff1a1bb
Successfully built langdetect
Installing collected packages: langdetect
Successfully installed langdetect-1.0.9
[ ]
from langdetect import detect
lang = detect(extractedText)
print(lang)  #detect the language 
en
[ ]
pip install translate #to translate text from one language to another
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting translate
  Downloading translate-3.6.1-py2.py3-none-any.whl (12 kB)
Requirement already satisfied: lxml in /usr/local/lib/python3.9/dist-packages (from translate) (4.9.2)
Collecting libretranslatepy==2.1.1
  Downloading libretranslatepy-2.1.1-py3-none-any.whl (3.2 kB)
Requirement already satisfied: click in /usr/local/lib/python3.9/dist-packages (from translate) (8.1.3)
Requirement already satisfied: requests in /usr/local/lib/python3.9/dist-packages (from translate) (2.27.1)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests->translate) (3.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests->translate) (2022.12.7)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests->translate) (2.0.12)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests->translate) (1.26.15)
Installing collected packages: libretranslatepy, translate
Successfully installed libretranslatepy-2.1.1 translate-3.6.1
[ ]
pip install gTTs #Google Text-to-Speech
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting gTTs
  Downloading gTTS-2.3.1-py3-none-any.whl (28 kB)
Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.9/dist-packages (from gTTs) (2.27.1)
Requirement already satisfied: click<8.2,>=7.1 in /usr/local/lib/python3.9/dist-packages (from gTTs) (8.1.3)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.9/dist-packages (from requests<3,>=2.27->gTTs) (2022.12.7)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.9/dist-packages (from requests<3,>=2.27->gTTs) (2.0.12)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.9/dist-packages (from requests<3,>=2.27->gTTs) (1.26.15)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.9/dist-packages (from requests<3,>=2.27->gTTs) (3.4)
Installing collected packages: gTTs
Successfully installed gTTs-2.3.1
```
pip install gtts
```
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

