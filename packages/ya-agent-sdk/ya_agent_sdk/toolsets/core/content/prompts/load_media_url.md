<load-media-url-tool>

<description>Load multimedia content (images, videos, audio) directly from URL for model analysis.</description>

<supported_urls>
{% if has_vision %}
- **Images**: `https://example.com/photo.jpg`, `https://example.com/image.png`
{% endif %}
{% if has_video %}
- **Videos**: `https://example.com/video.mp4`, `https://youtube.com/watch?v=xxx`
{% endif %}
{% if has_audio %}
- **Audio**: `https://example.com/audio.mp3`, `https://example.com/recording.wav`
{% endif %}
{% if has_document and enable_load_document %}
- **Documents**: `https://example.com/file.pdf`
{% endif %}
</supported_urls>

{% if not has_vision %}
<note>Image loading not supported. Use `view` tool instead.</note>
{% endif %}
{% if not has_video %}
<note>Video/YouTube loading not supported. Use `view` tool instead.</note>
{% endif %}
{% if not has_audio %}
<note>Audio loading not supported. Use `view` tool instead.</note>
{% endif %}

</load-media-url-tool>
