# camera_resolver.py
import socket
import base64
import ipaddress
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Timeouts pensados para Tailscale/WAN (evita falsos negativos)
DEFAULT_TIMEOUT_TCP = 1.2    # era 0.3
DEFAULT_TIMEOUT_RTS = 2.0    # era 0.5
MAX_WORKERS = 64
CACHE_FILE = ".cam_ip_cache.json"

def _basic_auth(user: str, password: str) -> str:
    token = f"{user}:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(token).decode("ascii")

def _rtsp_request(ip, port, user, password, path, timeout=DEFAULT_TIMEOUT_RTS) -> bool:
    """
    Abre TCP, envia RTSP OPTIONS e DESCRIBE; considera sucesso se houver "RTSP/1.0"
    (200 OK ou 401 Unauthorized também indicam IP válido p/ a câmera).
    """
    path = (path or "").lstrip("/")  # evita // no URL
    url = f"rtsp://{ip}:{port}/{path}"
    auth = _basic_auth(user, password)

    try:
        with socket.create_connection((ip, port), timeout=DEFAULT_TIMEOUT_TCP) as s:
            s.settimeout(timeout)

            cseq = 1
            req = (
                f"OPTIONS {url} RTSP/1.0\r\n"
                f"CSeq: {cseq}\r\n"
                f"Authorization: {auth}\r\n"
                f"User-Agent: pcounter/1.0\r\n"
                f"\r\n"
            ).encode("ascii")
            s.sendall(req)
            resp = s.recv(4096).decode(errors="ignore")
            if "RTSP/1.0" in resp:   # aceita 200/401/...
                return True

            cseq += 1
            req = (
                f"DESCRIBE {url} RTSP/1.0\r\n"
                f"CSeq: {cseq}\r\n"
                f"Accept: application/sdp\r\n"
                f"Authorization: {auth}\r\n"
                f"User-Agent: pcounter/1.0\r\n"
                f"\r\n"
            ).encode("ascii")
            s.sendall(req)
            resp = s.recv(8192).decode(errors="ignore")
            return "RTSP/1.0" in resp
    except Exception:
        return False

def _neighbors_first(all_ips, last_ip):
    last = ipaddress.ip_address(last_ip)
    ips = [ipaddress.ip_address(str(i)) for i in all_ips]
    near, far = [], []
    for ip in ips:
        (near if abs(int(ip) - int(last)) <= 10 else far).append(ip)
    near_sorted = sorted(near, key=lambda x: abs(int(x) - int(last)))
    return [str(ip) for ip in near_sorted] + [str(ip) for ip in far]

def _save_cache(ip: str):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"ip": ip, "ts": time.time()}, f)
    except Exception:
        pass

def _load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("ip")
    except Exception:
        return None

def resolve_rtsp(last_ip: str,
                 cidr: str,
                 user: str,
                 password: str,
                 path: str,
                 port: int = 554,
                 use_cache: bool = True) -> str | None:
    """
    1) tenta cache; 2) tenta last_ip; 3) varre /24 (vizinhos primeiro).
    Retorna 'rtsp://user:pass@IP:port/path' ou None.
    """
    path = (path or "").lstrip("/")

    if use_cache:
        cached = _load_cache()
        if cached and _rtsp_request(cached, port, user, password, path):
            return f"rtsp://{user}:{password}@{cached}:{port}/{path}"

    if last_ip and _rtsp_request(last_ip, port, user, password, path):
        if use_cache:
            _save_cache(last_ip)
        return f"rtsp://{user}:{password}@{last_ip}:{port}/{path}"

    network = ipaddress.ip_network(cidr, strict=False)
    candidates = [str(ip) for ip in network.hosts()]
    if last_ip and last_ip in candidates:
        candidates = _neighbors_first(candidates, last_ip)

    found_ip = None
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(_rtsp_request, ip, port, user, password, path): ip
            for ip in candidates
        }
        for fut in as_completed(futures):
            ip = futures[fut]
            ok = False
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                found_ip = ip
                break

    if found_ip:
        if use_cache:
            _save_cache(found_ip)
        return f"rtsp://{user}:{password}@{found_ip}:{port}/{path}"
    return None

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Resolve RTSP URL da câmera via varredura TCP/RTSP (WAN-friendly).")
    p.add_argument("--last", required=True)
    p.add_argument("--cidr", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--path", default="cam/realmonitor?channel=1&subtype=1")
    p.add_argument("--port", type=int, default=554)
    args = p.parse_args()
    url = resolve_rtsp(args.last, args.cidr, args.user, args.password, args.path, args.port)
    print(url or "")