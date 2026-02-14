<view-tool>
Read files from local filesystem. Supports text, images (PNG/JPEG/WebP), and videos (MP4/WebM/MOV).

<best-practices>
- For large files: use line_offset to read in chunks
- Increase line_limit if you need more context (default: 300)
- For PDF files: use `pdf_convert` tool instead
- Video files automatically use image understanding if model doesn't support video
</best-practices>
</view-tool>
