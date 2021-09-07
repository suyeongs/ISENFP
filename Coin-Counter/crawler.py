# pip install google_images_downlod
# pip install selenium

from google_images_download import google_images_download

def imageCrawling(keyword, dir):
    response = google_images_download.googleimagesdownload()

    arguments = {"keywords":keyword,        # 검색 키워드
                 "limit":100,               # 크롤링 이미지 수
                 "size":"large",            # 사이즈
#                 "format":"png",            # Image Format
                 "print_urls":True,         # 이미지 url 출력
                 "no_directory":True,
                 "chromedriver":"C:\\chromedriver_win32",
                 "output_directory":dir}    # 크롤링 이미지를 저장할 폴더

    paths = response.download(arguments)
    print(paths)

try:
    imageCrawling("koruna", 'C:\\result')
except:
    pass
