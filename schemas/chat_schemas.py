from fastapi import Form
from typing import Annotated

# Schemas for the chat-related routes. (Swagger UI)
Email = Annotated[str, Form(...)]
SessionId = Annotated[str, Form(...)]
Sender = Annotated[str, Form(...)]
MessageText = Annotated[str, Form(...)]
NewName = Annotated[str, Form(...)]
MessageIndex = Annotated[int, Form(...)]
NewText = Annotated[str, Form(...)]
DeleteSubsequent = Annotated[bool, Form(...)]
