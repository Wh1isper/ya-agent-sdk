<load-media-url-tool>
<best-practices>
- Use direct media URLs only when the active model supports that media type.
- Provide focused analysis instructions instead of asking for broad inspection.
</best-practices>
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
