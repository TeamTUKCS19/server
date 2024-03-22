import webbrowser
import os

# 클라이언트 HTML 파일 경로
client_html_path = 'client.html'

server_url = 'http://127.0.0.1:5000/upload_video'


def open_browser():
    # 클라이언트 HTML 파일을 웹 브라우저로 엽니다.
    webbrowser.open('file://' + os.path.realpath(client_html_path))


if __name__ == '__main__':
    open_browser()
