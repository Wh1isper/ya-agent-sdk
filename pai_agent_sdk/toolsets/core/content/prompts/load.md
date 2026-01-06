# Load Tool Usage Guide

Use this tool to load web content from HTTP/HTTPS URLs:

{% if has_vision %}
- **Images**: Returns visual content for direct model processing
{% else %}
- **Images**: Not supported by current model. Use `read_image` tool instead.
{% endif %}
{% if has_video %}
- **Videos**: Returns video content for direct model processing
{% else %}
- **Videos**: Not supported by current model. Use `read_video` tool instead.
{% endif %}
- **Audio**: Returns audio content
- **Text/HTML**: Returns document content for processing
{% if enable_load_document and has_document %}
- **Documents (PDF)**: Returns document content for model processing
{% else %}
- **Documents (PDF)**: Not supported. Use download tool to save locally, then use pdf_convert.
{% endif %}

**Note**: Only HTTP/HTTPS URLs are supported. Other protocols will be rejected.
