#!/bin/bash

WATCH_DIR="$(cd "$(dirname "$0")" && pwd)/videos"
echo "Monitoring directory: $WATCH_DIR"

# Define your email and subject
TO_EMAIL="shivomsharma13@hotmail.com"
SUBJECT="VIDEO UPLOADED FROM FILE"

# Start monitoring the directory
inotifywait -m -e create --format '%w%f' "$WATCH_DIR" | while read NEW_FILE
do
	# Check if the new file is a video (mp4, mkv, etc.)
	if [[ "$NEW_FILE" =~ \.(mp4|mkv|avi|mov)$ ]]; then
		# Send the email with the video attached, no message body
		mail -s "$SUBJECT" -a "$NEW_FILE" "$TO_EMAIL" < /dev/null
		echo "Sent email for $NEW_FILE"
	fi
done

