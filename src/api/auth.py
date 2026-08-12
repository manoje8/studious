from datetime import UTC, datetime, timedelta

import jwt
from fastapi import HTTPException, status
from pydantic import BaseModel

from src.api.config_api import config_api
from src.api.utils_api import verify_password


class TokenPayload(BaseModel):
    sub: str
    exp: datetime
    role: str = "user"
    metadata: dict = {}


class AuthHandler:
    def __init__(self):
        self.accounts = {}
        account_list = config_api.ACCOUNTS

        if account_list:
            for account in account_list.split(","):
                user_name, password = account.split(":", 1)
                if not user_name or not password:
                    raise ValueError
                self.accounts[user_name] = password

    def verify_password(self, user_name: str, plain_password: str) -> bool:
        if user_name not in self.accounts:
            return False

        stored_password = self.accounts[user_name]
        return verify_password(plain_password, stored_password)

    def create_access_token(
        self, user_name: str, role: str = "user", expire_hours: int = None, metadata: dict = None
    ) -> str:
        if not expire_hours:
            expire_hours = 5
        expire = datetime.now(UTC) + timedelta(hours=expire_hours)
        payload = TokenPayload(sub=user_name, role=role, exp=expire, metadata=metadata or {})

        return jwt.encode(payload.model_dump(), config_api.SECRET_KEY, config_api.ALGORITHM)

    def validate_token(self, token: str) -> dict:
        try:
            if config_api.ALGORITHM is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Invalid JWT algorithm",
                )

            payload = jwt.decode(token, config_api.SECRET_KEY, config_api.ALGORITHM)
            expire_timestamp = payload["exp"]
            expire_time = datetime.fromtimestamp(expire_timestamp, UTC)

            if datetime.now(UTC) > expire_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            return {
                "user_name": payload["sub"],
                "role": payload["role"],
                "metadata": payload["metadata"],
                "exp": expire_time,
            }
        except jwt.exceptions.PyJWTError as err:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            ) from err


auth_handler = AuthHandler()
