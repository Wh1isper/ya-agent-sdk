<load-tool>

<description>Load images or videos directly from URL for model analysis.</description>

<supported_urls>
{% if has_vision %}
- **Images**: `https://example.com/photo.jpg`, `https://example.com/image.png`
{% endif %}
{% if has_video %}
- **Videos**: `https://example.com/video.mp4`, `https://youtube.com/watch?v=xxx`
{% endif %}
{% if has_document and enable_load_document %}
- **Documents**: `https://example.com/file.pdf`
{% endif %}
</supported_urls>

{% if not has_vision %}
<note>Image loading not supported. Use `read_image` tool instead.</note>
{% endif %}
{% if not has_video %}
<note>Video/YouTube loading not supported. Use `read_video` tool instead.</note>
{% endif %}

</load-tool>
