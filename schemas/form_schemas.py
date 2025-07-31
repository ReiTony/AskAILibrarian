from fastapi import Form
from typing import Annotated


# fields for login
CardNumber = Annotated[str, Form(...)]
Password = Annotated[str, Form(...)]
