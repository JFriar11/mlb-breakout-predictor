"""Bounded ordinary HTTP access; stop on access restrictions and persist cooldowns."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import time
from urllib.parse import urlsplit
import requests


class AccessRestricted(RuntimeError):
    pass


def retry_after_seconds(value, now):
    if value:
        try:
            return max(0., float(value))
        except ValueError:
            try:
                return max(0., parsedate_to_datetime(value).timestamp()-now)
            except (ValueError, TypeError, OverflowError):
                pass
    return 60.


class HttpAccess:
    def __init__(self, cooldown_path):
        self.path = cooldown_path
        self.cooldowns = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.blocked = set()

    def request(self, original, session, method, url, **kwargs):
        host = urlsplit(url).netloc
        now = time.time()
        until = self.cooldowns.get(host, 0)
        if host in self.blocked or until > now:
            when = datetime.fromtimestamp(until, timezone.utc).isoformat()
            raise AccessRestricted(f'Access stopped for {host}; cooldown until at least {when}. No automatic retry.')
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (10, 90)
        response = original(session, method, url, **kwargs)
        if response.status_code in (401, 403, 429):
            self.blocked.add(host)
            if response.status_code == 429:
                until = now + retry_after_seconds(response.headers.get('Retry-After'), now)
                self.cooldowns[host] = until
                self.path.write_text(json.dumps(self.cooldowns, indent=2)+'\n')
            raise AccessRestricted(f'HTTP {response.status_code} from {host}; '
                                   f'Retry-After={response.headers.get("Retry-After")!r}. '
                                   'No further requests to this host in this run.')
        response.raise_for_status()
        return response
