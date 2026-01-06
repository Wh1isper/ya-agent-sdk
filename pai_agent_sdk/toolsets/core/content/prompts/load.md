# Load Tool Usage Guide

Use this tool to load web content from HTTP/HTTPS URLs:

- **Images**: Returns visual content (requires vision capability)
- **Videos**: Returns video content (requires video understanding)
- **Audio**: Returns audio content
- **Text/HTML**: Returns document content for processing
{% if enable_load_document %}
- **Documents (PDF)**: Returns document content for model processing
{% else %}
- **Documents (PDF)**: Not supported. Use download tool to save locally, then use pdf_convert.
{% endif %}

**Note**: Only HTTP/HTTPS URLs are supported. Other protocols will be rejected.
