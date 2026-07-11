from pydantic import BaseModel

class ImagePart(BaseModel):
    url: str | None = None
    data_base64: str | None = None

class Attachment(BaseModel):
    type: str
    url: str | None = None
    data_base64: str | None = None
