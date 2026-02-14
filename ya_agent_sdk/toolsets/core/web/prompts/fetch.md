<fetch-tool>
Read web files or check resource availability via HTTP.

<best-practices>
- Use head_only=True to check existence without downloading content
- Images are returned inline as BinaryContent for visual analysis
- For large files (>60K chars), content is truncated; use `download` instead
- For PDF files, download first then use `pdf_convert` tool
- Returns content_type, content_length, status_code for HEAD requests
</best-practices>
</fetch-tool>
