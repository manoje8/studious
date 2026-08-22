import bcrypt

HASH_PREFIX = "{bcrypt}"


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hash_pass = bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    return f"{HASH_PREFIX}{hash_pass}"


def verify_password(plain_password: str, stored_password: str) -> bool:
    if stored_password.startswith(HASH_PREFIX):
        hash_pass = stored_password[len(HASH_PREFIX) :]
        if not hash_pass:
            return False

        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hash_pass.encode("utf-8"))
        except ValueError:
            return False

    return plain_password == stored_password
