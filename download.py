from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import os, time, sys

# Set your own API Key and Secret Key
key = "XXXXXXXXXX"
secret = "XXXXXXXXXX"
wait_time = 0.5

keyword = sys.argv[1]
savedir = "./" + keyword

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = keyword,
    per_page = 300,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, license'
)

photos = result['photos']

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)
