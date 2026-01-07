<download-tool>
Download files from URLs and save to local filesystem.

<best-practices>
- Downloads multiple URLs in parallel for efficiency
- Files saved with UUID names; use `move` tool to rename if needed
- For PDF content, download first then use `pdf_convert` tool
- For web page content, use `scrape` tool instead
- For quick viewing without saving, use `fetch` tool
</best-practices>
</download-tool>
