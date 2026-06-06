"""Project package initialization.

Silence noisy `transformers` informational messages about
`LambdaRuntimeClient` by setting the library verbosity to ERROR.
This runs early on import for any module under `src`.
"""

import logging as _logging

try:
    from transformers import logging as _hf_logging

    _hf_logging.set_verbosity_error()
    _logging.getLogger("transformers").setLevel(_logging.ERROR)
except Exception:
    # If transformers isn't installed or something goes wrong, ignore.
    pass
