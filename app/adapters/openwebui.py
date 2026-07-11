from app.models.chat import ChatMessage


def extract_text_and_images(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    images: list[str] = []
    for msg in messages:
        if msg.role != 'user':
            continue
        content = msg.content
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                    elif part.get('type') == 'image_url':
                        image_url = part.get('image_url', {})
                        if isinstance(image_url, dict):
                            url = image_url.get('url')
                            if isinstance(url, str):
                                images.append(url)
    return '\n'.join([t for t in text_parts if t]).strip(), images
