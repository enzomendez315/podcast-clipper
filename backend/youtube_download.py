from pytubefix import YouTube
from pytubefix.cli import on_progress

url1 = "https://www.youtube.com/watch?v=j_TvWRS2_Hw"
url2 = "https://www.youtube.com/watch?v=Yvt9vlvuqSQ"

youtube = YouTube(url2, on_progress_callback=on_progress)
print(youtube.title)

youtube_stream = youtube.streams.get_highest_resolution()
youtube_stream.download()