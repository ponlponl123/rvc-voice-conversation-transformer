import os
import requests
from clint.textui import progress

class Downloader:
    def __init__(self, url: str, path: str = None, fileName: str = None, downloadOnReady: bool = False):
        self.url = url
        self.percent = 0
        self.downloaded = False
        self.path = path if path else "tmp"
        self.fileName = url.split('/')[-1] if not fileName else fileName
        self.outputFilename = fileName if fileName else self.fileName
        self.downloadOnReady = downloadOnReady
        if downloadOnReady:
            self.download()

    async def download(self):
        if not self.url:
            raise ValueError("URL is not set")
        r = requests.get(self.url, stream=True, timeout=10)
        path = f"tmp/{self.url.split('/')[-1]}"
        output = path
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not r.ok:
            raise ValueError(f"Error downloading file: {r.status_code}")
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
                if chunk:
                    self.percent = int((len(chunk) / total_length) * 100)
                    if self.percent > 100:
                        self.percent = 100
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        if self.path:
            output = os.path.join(os.getcwd(), self.path, self.outputFilename)
            os.rename(path, output)
        else:
            os.rename(path, output)
        self.percent = 100
        self.downloaded = True
        return output

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/ponlponl123/ponlponl123/main/README.md"
    downloader = Downloader(url, path="downloads")
    downloader.download()
    print(f"Downloaded file: {downloader.fileName}")
    print(f"Download path: {downloader.path}")
    print(f"Download percent: {downloader.percent}%")