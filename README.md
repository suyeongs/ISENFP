# ISENFP

# Money Detection
* Origin : [link](https://github.com/A3M4/Coin-Counter)

# Image Web Crawling
* Origin : [link](https://pino93.blog.me/221707621434)
## 전처리 이미지 모으기
1. 
```pip
$pip install google_images_download
$pip install selenium
```

2. 크롬 버전에 맞게 Chromedriver.exe 설치
- 버전 확인 : 우측 상단 메뉴 --> 도움말 --> Chrome 정보)
- 설치 링크 : https://chromedriver.chromium.org/downloads

3. 파이썬 파일 생성 (imgCrawler.py)
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
python imgCrawler.py
```
