<read-media-tool>

<description>Read an HTTP/HTTPS image, video, or audio URL by downloading bounded bytes and attaching them as binary media for model analysis.</description>

<usage>
- Use `read_media` for direct image, video, or audio URLs when the current model supports that media type.
- For public YouTube URLs, use `read_media` directly. Models with direct YouTube URL support, such as Gemini video-capable profiles, receive the YouTube URL without downloading it.
- Pass `instructions` for focused analysis such as OCR, UI inspection, transcription, timestamped summary, speaker identification, or extracting visual details.
- The tool reads media into memory only within configured inline limits and compresses images before attachment when the model image limit requires it.
- If the tool reports unsupported media, missing model capability, or size limits, use `download` to save the URL locally, compress or transcode it with shell tools if needed, then call `view` on the local path with focused `instructions`.
</usage>

</read-media-tool>
