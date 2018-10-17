# sbubby

Classify images by which subreddits they most likely belong to.

## Tips

Downloading a lot of images on a system with low memory (like a Raspberry Pi)? Try simplifying the .json first by piping it through `jq -c -M '[.[] | {id, url, subreddit}]'` to cut down on memory usage!
