from fastapi import Form
from typing import Annotated
from pydantic import EmailStr

# Schemas for the authentication-related routes. (Swagger UI)
Email = Annotated[EmailStr, Form(...)]
Username = Annotated[str, Form(...)]
Password = Annotated[str, Form(...)]
Code = Annotated[str, Form(...)]
EmailOrUsername = Annotated[str, Form(...)]
ResetToken = Annotated[str, Form(...)]
NewPassword = Annotated[str, Form(...)]