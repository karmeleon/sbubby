import os.path
import time

from PIL import Image
import praw

import common
from predict import Predictor

def main():
	reddit = login_to_reddit()

	me = reddit.user

	predictor = Predictor()

	print('Listening for comments')

	for comment in reddit.subreddit('sandboxtest').stream.comments():
		if '/u/sbubbot' in comment.body:
			# check to see if we've already replied to this comment
			already_replied = False
			for reply in comment.replies:
				if reply.author.name == me.name:
					already_replied = True
			if already_replied:
				print("found a comment we've already replied to")
				continue
			
			# pull the image
			print('pulling image')
			image = get_image_for_comment(comment)

			if image is None:
				print('posting failure comment')
				post_failure_comment(reddit, comment)
			else:
				print('predicting')
				start = time.time()
				results = predictor.predict(image)
				print('predicting took {} seconds, posting success comment'.format(time.time() - start))
				post_success_comment(reddit, comment, results)

def post_failure_comment(reddit, comment):
	message = "I couldn't find a valid image in this post's link :(" + get_footer()
	comment.reply(message)

def post_success_comment(reddit, comment, results):
	message = "Here's what subreddits I think this image could fit in with:\n\n"
	message += format_table_from_results(results)
	message += get_footer()

	comment.reply(message)

def format_table_from_results(results):
	table = 'subreddit | probability\n--- | ---\n'
	for result in results:
		# don't print out things that have < 1% similarity
		if result[1] < 0.01:
			continue
		table += '{} | {:.3f}\n'.format(*result)
	return table

def get_footer():
	return """
---
^([about](https://github.com/karmeleon/sbubby/blob/master/REDDIT_README.md))
"""

def get_image_for_comment(comment):
	link = comment.submission.url

	return common.fetch_image(link)

def login_to_reddit():
	if not os.path.isfile('praw.ini'):
		print("Couldn't find praw.ini in this directory! Try making one if this breaks: https://praw.readthedocs.io/en/latest/getting_started/configuration/prawini.html")
		exit(1)

	reddit = praw.Reddit('sbubbot', user_agent='python:subbot:v0.0.1 (by /u/notverycreative1)')
	print('Logged in!')
	return reddit

if __name__ == '__main__':
	main()
