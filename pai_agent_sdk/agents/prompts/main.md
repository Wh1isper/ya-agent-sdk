# Main Agent System Prompt

You are a helpful AI assistant.

{% if instructions %}
## Instructions

{{ instructions }}
{% endif %}

{% if capabilities %}
## Capabilities

{% for capability in capabilities %}
- {{ capability }}
{% endfor %}
{% endif %}
