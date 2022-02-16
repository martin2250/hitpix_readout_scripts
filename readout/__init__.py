from dataclasses import dataclass, field
import threading
from typing import Optional

@dataclass
class Response:
    data: Optional[bytes] = None
    event: threading.Event = field(default_factory=lambda: threading.Event())
    name: Optional[str] = None
