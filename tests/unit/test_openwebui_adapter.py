from app.adapters.openwebui import extract_text_and_images
from app.models.chat import ChatMessage


def test_extract_text_and_images_from_openai_content_parts():
    text, images = extract_text_and_images([
        ChatMessage(
            role='user',
            content=[
                {'type': 'text', 'text': 'describe this'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}},
            ],
        ),
        ChatMessage(role='assistant', content='ignored'),
    ])

    assert text == 'describe this'
    assert images == ['data:image/png;base64,abc']
