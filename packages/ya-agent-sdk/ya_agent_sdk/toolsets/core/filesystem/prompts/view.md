<view-tool>
Read files from local filesystem. Supports text, images (PNG/JPEG/WebP), videos (MP4/WebM/MOV), and audio (MP3/WAV/OGG).

<best-practices>
- For large files: use line_offset to read in chunks
- Increase line_limit if you need more context (default: 300)
- For PDF files: use `pdf_convert` tool instead
- For image, video, or audio files, pass `instructions` when you need focused analysis such as OCR, transcription, timestamped review, UI QA, speaker labels, or extracting specific details
- Use multiple `view` calls with narrower `instructions` when a previous media result mentions unclear, low-confidence, omitted, summarized, or high-detail regions
- Ask for the analyzer to name omitted details, uncertain observations, and useful follow-up focuses when you need complete media understanding
- Video and audio files automatically use a fallback understanding agent when the active model lacks the matching media capability
</best-practices>
</view-tool>
