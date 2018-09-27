# ml-irl

Attempt to predict how well an image post will do on a subreddit using TensorFlow

## Tips

Downloading a lot of images on a system with low memory (like a Raspberry Pi)? Try simplifying the .json first by piping it through `jq -c -M '[.[] | {id, url}]'` to cut down on memory usage!
