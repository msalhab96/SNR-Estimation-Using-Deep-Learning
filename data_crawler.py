import re
from time import sleep
from bs4 import BeautifulSoup
import requests
import os 


def download_audio(save_to: str, url: str):
    content = requests.get(url).content
    with open(save_to, 'wb') as f:
        f.write(content)

def request_page(url: str, page: int):
    try:
        return requests.get(
            url=url,
            params={
                'page': page
            }
        )
    except:
        return None

def crawl_voices(save_to):
    # The voice assistant page because it does not contain too much music in the background 
    base_url = 'https://www.voices.com/hire/voice-over/skill/voice-assistant'
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    counter = 0
    total = 0
    while True:
        request = request_page(base_url, counter)
        if request is None:
            continue
        soup = BeautifulSoup(request.content)
        all_audios = soup.find_all('audio')
        total += len(all_audios)
        if len(all_audios) == 0:
            break
        for audio in all_audios:
            audio_source = audio.find('source')
            try:
                audio_url = audio_source.get('data-src')
            except:
                sleep(2)
                continue
            path = os.path.join(save_to, audio_url.split('/')[-1] + '.mp3')
            download_audio(path, audio_url)
        counter += 1
        print(f'total found audios is {total}')

def crawl_bunny_studio(save_to: str):
    base_url = 'https://bunnystudio.com/voice/search/'
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    counter = 3
    total = 0
    while True:
        request = request_page(base_url, counter)
        if request is None:
            continue
        content = str(request.content)
        all_audios = re.findall(
            'sample/Bunny Studio Voice_-_ID_[\da-zA-Z]+_-_Sample_\d+\.mp3', 
            content
            )
        total += len(all_audios)
        if len(all_audios) == 0:
            break
        for audio in all_audios:
            audio_url = 'https://voicebunny.s3.amazonaws.com/' + audio
            path = os.path.join(save_to, audio.split('_')[-1])
            download_audio(path, audio_url)
        counter += 1
        print(f'total found audios is {total}')

if __name__ == '__main__':
    crawl_voices('data')
    crawl_bunny_studio('bunny')
