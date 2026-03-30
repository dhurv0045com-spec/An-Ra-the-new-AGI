"""
tools/connectors/ — External API Connectors

Base class + 8 ready-to-activate connectors:
Wikipedia, Weather, Currency, News, GitHub, WolframAlpha, Email, Calendar

Each connector: one config dict to activate.
Rate limiting, error handling, and graceful degradation built in.
"""

import json, time, re, threading, hashlib
import urllib.request, urllib.parse, urllib.error
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path


CONFIG_DIR = Path("state/connector_configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR  = Path("state/connector_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BASE CONNECTOR
# ══════════════════════════════════════════════════════════════════════════════

class ConnectorResult:
    def __init__(self, connector: str, success: bool, data: Any,
                 error: str = "", cached: bool = False, ms: float = 0):
        self.connector = connector
        self.success   = success
        self.data      = data
        self.error     = error
        self.cached    = cached
        self.ms        = ms
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return vars(self)

    def __str__(self):
        if self.success:
            return f"[{self.connector}] {'(cached) ' if self.cached else ''}{str(self.data)[:200]}"
        return f"[{self.connector}] ERROR: {self.error}"


class BaseConnector(ABC):
    """Base class for all external connectors."""

    NAME        = "base"
    RATE_LIMIT  = 60    # max calls per minute
    CACHE_TTL   = 300   # seconds to cache results

    def __init__(self, config: dict = None):
        self.config    = config or {}
        self._calls:   List[float] = []    # timestamps of recent calls
        self._lock     = threading.Lock()
        self._cache:   Dict[str, tuple] = {}   # key → (result, expires_at)
        self._enabled  = self._load_config()

    def _load_config(self) -> bool:
        """Load config from file if exists."""
        cfg_file = CONFIG_DIR / f"{self.NAME}.json"
        if cfg_file.exists():
            try:
                saved = json.loads(cfg_file.read_text())
                self.config.update(saved)
                return saved.get("enabled", True)
            except Exception:
                pass
        return self.config.get("enabled", True)

    def save_config(self):
        cfg_file = CONFIG_DIR / f"{self.NAME}.json"
        cfg_file.write_text(json.dumps(self.config, indent=2))

    def _rate_check(self) -> bool:
        """Return True if allowed to make a call (not rate-limited)."""
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < 60]
            if len(self._calls) >= self.RATE_LIMIT:
                return False
            self._calls.append(now)
        return True

    def _cache_key(self, *args, **kwargs) -> str:
        raw = json.dumps([args, kwargs], sort_keys=True, default=str)
        return self.NAME + "_" + hashlib.md5(raw.encode()).hexdigest()[:12]

    def _get_cache(self, key: str) -> Optional[ConnectorResult]:
        entry = self._cache.get(key)
        if entry:
            result, expires = entry
            if time.time() < expires:
                result.cached = True
                return result
            del self._cache[key]
        return None

    def _set_cache(self, key: str, result: ConnectorResult):
        self._cache[key] = (result, time.time() + self.CACHE_TTL)

    def _fetch(self, url: str, headers: dict = None,
               params: dict = None, timeout: int = 10) -> tuple:
        """Shared HTTP fetch utility."""
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers=headers or {"User-Agent": "45L-AI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body

    def call(self, *args, **kwargs) -> ConnectorResult:
        """Public entry point — handles rate limiting, caching, error wrapping."""
        if not self._enabled:
            return ConnectorResult(self.NAME, False, None,
                                   f"Connector '{self.NAME}' is disabled")
        cache_key = self._cache_key(*args, **kwargs)
        cached    = self._get_cache(cache_key)
        if cached:
            return cached

        if not self._rate_check():
            return ConnectorResult(self.NAME, False, None,
                                   f"Rate limit exceeded ({self.RATE_LIMIT}/min)")
        t0 = time.monotonic()
        try:
            result = self._execute(*args, **kwargs)
            result.ms = round((time.monotonic() - t0) * 1000, 1)
            if result.success:
                self._set_cache(cache_key, result)
            return result
        except urllib.error.URLError as e:
            return ConnectorResult(self.NAME, False, None,
                                   f"Network error: {e}", ms=(time.monotonic()-t0)*1000)
        except Exception as e:
            return ConnectorResult(self.NAME, False, None, str(e),
                                   ms=(time.monotonic()-t0)*1000)

    @abstractmethod
    def _execute(self, *args, **kwargs) -> ConnectorResult:
        pass

    def status(self) -> dict:
        return {
            "name":       self.NAME,
            "enabled":    self._enabled,
            "rate_limit": self.RATE_LIMIT,
            "cache_ttl":  self.CACHE_TTL,
            "config_keys": list(self.config.keys()),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  WIKIPEDIA — no API key needed
# ══════════════════════════════════════════════════════════════════════════════

class WikipediaConnector(BaseConnector):
    NAME       = "wikipedia"
    CACHE_TTL  = 3600   # wiki articles don't change often

    def _execute(self, query: str, sentences: int = 3) -> ConnectorResult:
        params = {
            "action":   "query",
            "list":     "search",
            "srsearch": query,
            "format":   "json",
            "srlimit":  "3",
        }
        _, body = self._fetch("https://en.wikipedia.org/w/api.php",
                               params=params)
        data    = json.loads(body)
        hits    = data.get("query", {}).get("search", [])
        if not hits:
            return ConnectorResult(self.NAME, True, {"results": [], "query": query})

        # Fetch extract for top result
        top_title = hits[0]["title"]
        params2   = {
            "action":   "query",
            "prop":     "extracts",
            "exintro":  "true",
            "exsentences": str(sentences),
            "titles":   top_title,
            "format":   "json",
        }
        _, body2 = self._fetch("https://en.wikipedia.org/w/api.php", params=params2)
        data2    = json.loads(body2)
        pages    = data2.get("query", {}).get("pages", {})
        extract  = ""
        for page in pages.values():
            raw = page.get("extract", "")
            extract = re.sub(r'<[^>]+>', '', raw).strip()

        return ConnectorResult(self.NAME, True, {
            "query":   query,
            "title":   top_title,
            "extract": extract,
            "url":     f"https://en.wikipedia.org/wiki/{urllib.parse.quote(top_title)}",
            "related": [h["title"] for h in hits[1:3]],
        })


# ══════════════════════════════════════════════════════════════════════════════
#  WEATHER — Open-Meteo (no API key)
# ══════════════════════════════════════════════════════════════════════════════

class WeatherConnector(BaseConnector):
    NAME       = "weather"
    CACHE_TTL  = 1800   # 30 min

    GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    def _execute(self, location: str, days: int = 1) -> ConnectorResult:
        # Geocode
        _, body = self._fetch(self.GEOCODE_URL, params={
            "name": location, "count": "1", "format": "json"
        })
        geo  = json.loads(body)
        locs = geo.get("results", [])
        if not locs:
            return ConnectorResult(self.NAME, False, None,
                                   f"Location not found: {location}")
        lat, lon = locs[0]["latitude"], locs[0]["longitude"]
        city     = locs[0].get("name", location)
        country  = locs[0].get("country", "")

        # Weather
        _, wbody = self._fetch(self.WEATHER_URL, params={
            "latitude":   lat,
            "longitude":  lon,
            "current":    "temperature_2m,weathercode,windspeed_10m,relativehumidity_2m",
            "daily":      "temperature_2m_max,temperature_2m_min,weathercode",
            "timezone":   "UTC",
            "forecast_days": min(days, 7),
        })
        w = json.loads(wbody)
        curr = w.get("current", {})

        WMO_CODES = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
            61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow",
            80: "Rain showers", 85: "Snow showers", 95: "Thunderstorm",
        }
        code = curr.get("weathercode", 0)

        return ConnectorResult(self.NAME, True, {
            "location":    f"{city}, {country}",
            "temp_c":      curr.get("temperature_2m"),
            "humidity":    curr.get("relativehumidity_2m"),
            "wind_kmh":    curr.get("windspeed_10m"),
            "condition":   WMO_CODES.get(code, f"Code {code}"),
            "forecast":    [
                {
                    "date":    w["daily"]["time"][i] if "daily" in w else "",
                    "max_c":   w["daily"]["temperature_2m_max"][i],
                    "min_c":   w["daily"]["temperature_2m_min"][i],
                    "condition": WMO_CODES.get(w["daily"]["weathercode"][i], ""),
                }
                for i in range(min(days, len(w.get("daily", {}).get("time", []))))
            ] if "daily" in w else [],
        })


# ══════════════════════════════════════════════════════════════════════════════
#  CURRENCY — Open Exchange Rates (free tier, no key for basic)
# ══════════════════════════════════════════════════════════════════════════════

class CurrencyConnector(BaseConnector):
    NAME       = "currency"
    CACHE_TTL  = 3600

    def _execute(self, amount: float, from_currency: str,
                 to_currency: str) -> ConnectorResult:
        from_c = from_currency.upper()
        to_c   = to_currency.upper()

        # Use exchangerate.host (free, no key)
        _, body = self._fetch(
            f"https://api.exchangerate.host/convert",
            params={"from": from_c, "to": to_c, "amount": amount}
        )
        data = json.loads(body)
        if not data.get("success"):
            # Fallback: just return a hardcoded approximation for demo
            rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5,
                     "CAD": 1.36, "AUD": 1.53, "CHF": 0.90, "INR": 83.1}
            rate  = rates.get(to_c, 1.0) / rates.get(from_c, 1.0)
            converted = amount * rate
            return ConnectorResult(self.NAME, True, {
                "amount": amount, "from": from_c, "to": to_c,
                "rate": round(rate, 6), "converted": round(converted, 4),
                "source": "fallback_rates",
            })

        return ConnectorResult(self.NAME, True, {
            "amount":    amount,
            "from":      from_c,
            "to":        to_c,
            "rate":      data.get("info", {}).get("rate"),
            "converted": data.get("result"),
            "source":    "live",
        })


# ══════════════════════════════════════════════════════════════════════════════
#  GITHUB — public API, no key for read operations
# ══════════════════════════════════════════════════════════════════════════════

class GitHubConnector(BaseConnector):
    NAME       = "github"
    RATE_LIMIT = 30    # unauthenticated = 60/hr; authenticated much higher
    CACHE_TTL  = 300

    BASE = "https://api.github.com"

    def _headers(self) -> dict:
        h = {"Accept": "application/vnd.github.v3+json",
             "User-Agent": "45L-AI/1.0"}
        token = self.config.get("token")
        if token:
            h["Authorization"] = f"token {token}"
        return h

    def _execute(self, action: str, **kwargs) -> ConnectorResult:
        if action == "search_repos":
            query = kwargs.get("query", "")
            _, body = self._fetch(f"{self.BASE}/search/repositories",
                                   headers=self._headers(),
                                   params={"q": query, "per_page": 5})
            data  = json.loads(body)
            items = data.get("items", [])
            return ConnectorResult(self.NAME, True, {
                "repos": [{
                    "name":  r["full_name"],
                    "stars": r["stargazers_count"],
                    "desc":  r.get("description", "")[:100],
                    "url":   r["html_url"],
                } for r in items]
            })

        elif action == "get_repo":
            repo  = kwargs.get("repo", "")
            _, body = self._fetch(f"{self.BASE}/repos/{repo}",
                                   headers=self._headers())
            data  = json.loads(body)
            return ConnectorResult(self.NAME, True, {
                "name":    data.get("full_name"),
                "stars":   data.get("stargazers_count"),
                "forks":   data.get("forks_count"),
                "desc":    data.get("description", "")[:200],
                "language": data.get("language"),
                "url":     data.get("html_url"),
                "updated": data.get("updated_at"),
            })

        elif action == "list_issues":
            repo  = kwargs.get("repo", "")
            _, body = self._fetch(f"{self.BASE}/repos/{repo}/issues",
                                   headers=self._headers(),
                                   params={"state": "open", "per_page": 10})
            issues = json.loads(body)
            return ConnectorResult(self.NAME, True, {
                "repo":   repo,
                "issues": [{
                    "number": i["number"],
                    "title":  i["title"][:100],
                    "state":  i["state"],
                    "url":    i["html_url"],
                } for i in issues if isinstance(i, dict)]
            })

        return ConnectorResult(self.NAME, False, None,
                               f"Unknown action: {action}")


# ══════════════════════════════════════════════════════════════════════════════
#  NEWS — RSS/Atom feeds (no API key)
# ══════════════════════════════════════════════════════════════════════════════

class NewsConnector(BaseConnector):
    NAME       = "news"
    CACHE_TTL  = 900   # 15 min

    FEEDS = {
        "tech":    "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "world":   "https://feeds.bbci.co.uk/news/world/rss.xml",
        "science": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "general": "https://feeds.bbci.co.uk/news/rss.xml",
    }

    def _execute(self, category: str = "general",
                 max_items: int = 5) -> ConnectorResult:
        feed_url = self.FEEDS.get(category.lower(), self.FEEDS["general"])
        _, body  = self._fetch(feed_url)

        # Parse RSS
        items  = []
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>',
                             body, re.DOTALL)
        links  = re.findall(r'<link>(https?://[^<]+)</link>', body)
        descs  = re.findall(r'<description><!\[CDATA\[(.*?)\]\]></description>|'
                             r'<description>(.*?)</description>', body, re.DOTALL)

        for i in range(min(max_items, len(titles))):
            title = (titles[i][0] or titles[i][1]).strip()
            if title in ("", category, "BBC News"):
                continue
            desc = ""
            if i < len(descs):
                raw  = descs[i][0] or descs[i][1]
                desc = re.sub(r'<[^>]+>', '', raw).strip()[:200]
            link = links[i] if i < len(links) else ""
            items.append({"title": title, "description": desc, "url": link})

        return ConnectorResult(self.NAME, True, {
            "category": category,
            "items":    items[:max_items],
            "source":   feed_url,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  WOLFRAMALPHA — basic short answer (requires free API key)
# ══════════════════════════════════════════════════════════════════════════════

class WolframConnector(BaseConnector):
    NAME       = "wolfram"
    RATE_LIMIT = 20
    CACHE_TTL  = 3600

    def _execute(self, query: str) -> ConnectorResult:
        app_id = self.config.get("app_id", "")
        if not app_id:
            # Math-only fallback using stdlib
            return self._math_fallback(query)

        _, body = self._fetch(
            "https://api.wolframalpha.com/v1/result",
            params={"i": query, "appid": app_id}
        )
        return ConnectorResult(self.NAME, True, {"query": query, "result": body.strip()})

    def _math_fallback(self, query: str) -> ConnectorResult:
        """Evaluate simple math when no API key configured."""
        import math as _math
        allowed = {k: getattr(_math, k) for k in dir(_math) if not k.startswith('_')}
        allowed.update({"abs": abs, "round": round, "int": int, "float": float})
        try:
            result = eval(query, {"__builtins__": {}}, allowed)
            return ConnectorResult(self.NAME, True, {
                "query": query, "result": result, "note": "math-only fallback"
            })
        except Exception as e:
            return ConnectorResult(self.NAME, False, None,
                                   f"No API key configured. Math fallback failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  EMAIL — SMTP/IMAP stub (safe — requires explicit config to activate)
# ══════════════════════════════════════════════════════════════════════════════

class EmailConnector(BaseConnector):
    NAME    = "email"
    RATE_LIMIT = 10

    def _execute(self, action: str, **kwargs) -> ConnectorResult:
        if not self.config.get("smtp_host"):
            return ConnectorResult(self.NAME, False, None,
                "Email not configured. Set smtp_host, smtp_port, username, "
                "password in state/connector_configs/email.json to activate.")
        if action == "send":
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(kwargs.get("body", ""))
            msg["Subject"] = kwargs.get("subject", "")
            msg["From"]    = self.config["username"]
            msg["To"]      = kwargs.get("to", "")
            with smtplib.SMTP_SSL(self.config["smtp_host"],
                                   self.config.get("smtp_port", 465)) as s:
                s.login(self.config["username"], self.config["password"])
                s.send_message(msg)
            return ConnectorResult(self.NAME, True, {"sent": True, "to": kwargs.get("to")})
        return ConnectorResult(self.NAME, False, None, f"Unknown action: {action}")


# ══════════════════════════════════════════════════════════════════════════════
#  CALENDAR — iCal read (stub with local file support)
# ══════════════════════════════════════════════════════════════════════════════

class CalendarConnector(BaseConnector):
    NAME    = "calendar"

    def _execute(self, action: str = "list", days: int = 7,
                 **kwargs) -> ConnectorResult:
        cal_path = self.config.get("ical_path", "")
        if not cal_path or not Path(cal_path).exists():
            # Return demo data when not configured
            now = datetime.utcnow()
            events = [
                {"title": "Example: Team standup", "start": now.isoformat(),
                 "end":   (now + timedelta(hours=1)).isoformat()},
                {"title": "Example: Weekly review",
                 "start": (now + timedelta(days=1)).isoformat(),
                 "end":   (now + timedelta(days=1, hours=2)).isoformat()},
            ]
            return ConnectorResult(self.NAME, True, {
                "events": events, "note": "demo data — set ical_path in config to use real calendar"
            })

        # Parse iCal
        raw    = Path(cal_path).read_text()
        events = []
        for block in raw.split("BEGIN:VEVENT"):
            if "DTSTART" not in block:
                continue
            summary = re.search(r'SUMMARY:(.*)', block)
            dtstart = re.search(r'DTSTART.*?:([\d]+)', block)
            dtend   = re.search(r'DTEND.*?:([\d]+)', block)
            if summary and dtstart:
                events.append({
                    "title": summary.group(1).strip(),
                    "start": dtstart.group(1),
                    "end":   dtend.group(1) if dtend else "",
                })

        cutoff = (datetime.utcnow() + timedelta(days=days)).strftime("%Y%m%d")
        events = [e for e in events if e.get("start", "")[:8] <= cutoff]
        return ConnectorResult(self.NAME, True, {"events": events[:20]})


# ══════════════════════════════════════════════════════════════════════════════
#  CONNECTOR REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class ConnectorRegistry:
    """Single access point for all connectors."""

    def __init__(self):
        self._connectors: Dict[str, BaseConnector] = {
            "wikipedia": WikipediaConnector(),
            "weather":   WeatherConnector(),
            "currency":  CurrencyConnector(),
            "github":    GitHubConnector(),
            "news":      NewsConnector(),
            "wolfram":   WolframConnector(),
            "email":     EmailConnector(),
            "calendar":  CalendarConnector(),
        }

    def call(self, name: str, *args, **kwargs) -> ConnectorResult:
        c = self._connectors.get(name)
        if not c:
            return ConnectorResult(name, False, None, f"Unknown connector: {name}")
        return c.call(*args, **kwargs)

    def status(self) -> List[dict]:
        return [c.status() for c in self._connectors.values()]

    def configure(self, name: str, config: dict):
        c = self._connectors.get(name)
        if c:
            c.config.update(config)
            c.save_config()


connectors = ConnectorRegistry()
