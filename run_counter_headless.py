# run_counter_headless.py
import os, sys, subprocess, tempfile, shutil, argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
COUNTER_FILE = PROJECT_DIR / "line_counter_headless.py"  # usa a versão headless
MODEL_FILE   = PROJECT_DIR / "yolov8n.pt"                # opcional
LINE_CONFIG  = PROJECT_DIR / "line_config.json"          # essencial

# Defaults (podem ser sobrescritos por variáveis de ambiente ou args)
DEF_LAST_IP = os.getenv("CAM_LAST_IP", "192.168.1.2")
DEF_CIDR    = os.getenv("CAM_CIDR",    "192.168.1.0/24")
DEF_USER    = os.getenv("CAM_USER",    "admin")
DEF_PASS    = os.getenv("CAM_PASS",    "111229")
DEF_PATH    = os.getenv("CAM_PATH",    "cam/realmonitor?channel=1&subtype=1")
DEF_PORT    = int(os.getenv("CAM_PORT", "554"))

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-ip", default=DEF_LAST_IP)
    parser.add_argument("--cidr",    default=DEF_CIDR)
    parser.add_argument("--user",    default=DEF_USER)
    parser.add_argument("--password",default=DEF_PASS)
    parser.add_argument("--path",    default=DEF_PATH)
    parser.add_argument("--port",    type=int, default=DEF_PORT)
    args = parser.parse_args()

    # Resolver do RTSP automático (cache + varredura WAN-friendly)
    from camera_resolver import resolve_rtsp

    print("[1/3] Descobrindo RTSP válido (WAN/Tailscale-friendly)…")
    rtsp_url = resolve_rtsp(
        last_ip=args.last_ip,
        cidr=args.cidr,
        user=args.user,
        password=args.password,
        path=args.path,
        port=args.port,
        use_cache=True
    )

    if not rtsp_url:
        rtsp_url = f"rtsp://{args.user}:{args.password}@{args.last_ip}:{args.port}/{args.path}"
        print("[AVISO] Resolver não confirmou IP por RTSP. Usando fallback:", rtsp_url)
    else:
        print("   ✓ RTSP encontrado:", rtsp_url)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print("[2/3] Preparando cópia temporária do line_counter_headless.py…")
        shutil.copy2(COUNTER_FILE, tmpdir / COUNTER_FILE.name)

        if LINE_CONFIG.exists():
            shutil.copy2(LINE_CONFIG, tmpdir / LINE_CONFIG.name)
        else:
            print(f"[AVISO] {LINE_CONFIG} não encontrado. O contador irá reclamar.")

        if MODEL_FILE.exists():
            shutil.copy2(MODEL_FILE, tmpdir / MODEL_FILE.name)

        env = os.environ.copy()
        env["RTSP_URL"] = rtsp_url  # lido pelo line_counter_headless
        env["HEADLESS"] = "1"
        env["HEADLESS_MAX_SEC"] = os.getenv("HEADLESS_MAX_SEC", "3600")  # ~1h
        env["RUN_TZ"] = "America/Sao_Paulo"
        env["REPORT_MODE"] = os.getenv("REPORT_MODE", "graph_excel")

        print("[3/3] Executando o contador (cópia temporária)…")
        print("Usando RTSP_URL =", rtsp_url)
        subprocess.run([sys.executable, str(tmpdir / COUNTER_FILE.name)],
                       cwd=str(tmpdir), env=env, check=False)

if __name__ == "__main__":
    run()