# Coin Counter
# ISENFP

# Image reprocessing
* Origin : [link](https://pino93.blog.me/221707621434)
## Image Web Crawling
1. 파이썬 모듈 설치
```pip
$pip install google_images_download
$pip install selenium
```

2. 크롬 버전에 맞게 Chromedriver.exe 설치
- 버전 확인 : 우측 상단 메뉴 → 도움말 → Chrome 정보)
- 설치 링크 : https://chromedriver.chromium.org/downloads

3. 파이썬 파일 생성 (crawler.py)
```python
from google_images_download import google_images_download

def imageCrawling(keyword, dir):
    response = google_images_download.googleimagesdownload()

    arguments = {"keywords":keyword,        # 검색 키워드
                 "limit":100,               # 크롤링 이미지 수
                 "size":"large",            # 사이즈
#                 "format":"png",            # Image Format
                 "print_urls":True,         # 이미지 url 출력
                 "no_directory":True,
                 "chromedriver":"(chromedriver.exe가 있는 폴더 경로)",  # 경로 입력 오류시 / --> //
                 "output_directory":dir}    # 크롤링 이미지를 저장할 폴더

    paths = response.download(arguments)
    print(paths)

try:
    imageCrawling("(검색어)", '(이미지를 저장할 폴더 경로)')  # 경로 입력 오류시 / --> //
except:
    pass

```
4. 파일 실행
```pip
$python crawler.py
```


# Money Detection
* Origin : [Coin-Counter](https://github.com/A3M4/Coin-Counter)
## Requirement
1. ISENFP repository 를 git clone 한다.
```
git clone https://github.com/suyeongs/ISENFP.git
```

2. Tensorflow models 를 git clone 하고, 환경을 셋팅한다.
```
git clone https://github.com/tensorflow/models.git
```
- 환경 셋팅 참고 링크 : [Tensorflow Models Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- 위의 링크에서 pip 말고, pip3으로 설치를 진행한다.
- 위의 링크에서 COCO API installation를 하던 중 make 에서 에러가 나면 cat Makefile 로 스크립트 하나씩 실행한다. 여기서 또한 python 말고, python3으로 고쳐서 실행한다.

3. ssd_inception_v2_coco tar파일을 다운받아서 Coin-Counter 폴더에 넣고 압축을 푼다.
- 다운로드 및 설명 링크 : [ssd_inception_v2_coco]
(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

4. OpenCV 라이브러리를 설치한다.
- [ubuntu18.04에서 opencv 설치하기](https://webnautes.tistory.com/1186)

5. 기타 필요한 라이브러리를 설치한다.
```
pip3 install pandas
```

## How To Use
1. Labelling and Create Dataset
    * training_data, test_data에 image파일을 적절히 분배하여 넣는다.
    * [LabelImg](https://tzutalin.github.io/labelImg/)를 이용하여 각 image들에 있는 money label 정보가 담긴 xml파일들을 얻는다.
    * xml 파일을 각 training_data, test_data 폴더에 옮겨 놓는다. (한 image와 한 xml파일이 한 쌍이다.)
    * 1_xml_to_csv.py 를 실행시킨 후 2_generate_tfrecords.py를 실행시킨다.
```
python3 1_xml_to_csv.py
python3 2_generate_tfrecords.py
```
2. Training the Model
    * ssd_inception_v2_coco.config 파일에서 num_classes, num_steps, batch_size 등의 값들을 알맞게 수정한다.
* 3_train.py 를 실행시킨다.
```
3. 

python3 4_export_inference_graph.py
```

## Future Improvements
* 크롤링한 이미지들을 전체적으로 resize한 후에 라벨링 작업 하는 것을 추천한다.
```
python resize_images.py --target-size (400,250)
```


# Raspberry PI Web Server
