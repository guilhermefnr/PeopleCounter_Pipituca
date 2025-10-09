# reporter.py
import os
import csv
import datetime as dt
import requests
import msal

# Fuso fixo GMT-3 (sem DST). Se quiser precisão total, use zoneinfo/pytz.
def _now_local():
    return dt.datetime.utcnow() - dt.timedelta(hours=3)

def _col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

# Colunas:
# A = Data
# 09..21 = B..N (13 colunas)
# TotalEntradas = O
DATA_COL = 1
FIRST_HOUR = 9
LAST_HOUR = 21
FIRST_HOUR_COL = DATA_COL + 1          # 2 -> B
LAST_HOUR_COL  = FIRST_HOUR_COL + (LAST_HOUR - FIRST_HOUR)  # 14 -> N
TOTAL_COL = LAST_HOUR_COL + 1          # 15 -> O

def _row_for_day(day: int) -> int:
    return 1 + day  # linha 2 = dia 1

def _col_for_hour(hour: int) -> int:
    if hour < FIRST_HOUR or hour > LAST_HOUR:
        raise ValueError("Hora fora da janela [9..21]")
    return FIRST_HOUR_COL + (hour - FIRST_HOUR)

def _headers():
    return ["Data"] + [f"{h:02d}" for h in range(FIRST_HOUR, LAST_HOUR + 1)] + ["TotalEntradas"]

class Reporter:
    def __init__(self):
        self.mode = os.getenv("REPORT_MODE", "graph_excel")  # "graph_excel" | "csv"
        # Credenciais/IDs para Graph
        self.tenant_id  = os.getenv("EXCEL_TENANT_ID")
        self.client_id  = os.getenv("EXCEL_CLIENT_ID")
        self.client_sec = os.getenv("EXCEL_CLIENT_SECRET")
        self.drive_id   = os.getenv("EXCEL_DRIVE_ID")   # OneDrive/SharePoint
        self.item_id    = os.getenv("EXCEL_ITEM_ID")    # ID do arquivo .xlsx
        self.worksheet  = os.getenv("EXCEL_WORKSHEET", "Planilha1")
        # CSV fallback
        self.csv_path   = os.getenv("CSV_PATH", "movimento_entradas.csv")

    # ===================== Excel (Graph) =====================
    def _get_graph_session(self):
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=authority,
            client_credential=self.client_sec
        )
        scopes = ["https://graph.microsoft.com/.default"]
        token = app.acquire_token_for_client(scopes=scopes)
        if "access_token" not in token:
            raise RuntimeError(f"Falha MSAL: {token.get('error_description')}")
        sess = requests.Session()
        sess.headers.update({"Authorization": f"Bearer {token['access_token']}"})
        return sess

    def _graph_base_url(self):
        return f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/items/{self.item_id}"

    def _ensure_month_header(self, session, base_url):
        headers = _headers()
        # Cabeçalho: linha 1 de A1 até última coluna (TotalEntradas)
        last_col_letter = _col_letter(TOTAL_COL)  # "O"
        url = f"{base_url}/workbook/worksheets/{self.worksheet}/range(address='A1:{last_col_letter}1')/values"
        session.patch(url, json={"values": [headers]})

    # ===================== Escrita por hora =====================
    def write_hourly(self, entradas_count: int, when: dt.datetime | None = None):
        if when is None:
            when = _now_local()

        day  = when.day
        hour = when.hour
        date_str = when.strftime("%Y-%m-%d")

        # Regras: grava SOMENTE às 10..21, cada ponto representa a janela anterior
        if hour < 10 or hour > 21:
            return False

        if self.mode == "graph_excel":
            sess = self._get_graph_session()
            base = self._graph_base_url()

            # Cabeçalho idempotente
            self._ensure_month_header(sess, base)

            row = _row_for_day(day)

            # Coluna de Data (A)
            a_addr = f"A{row}:A{row}"
            sess.patch(
                f"{base}/workbook/worksheets/{self.worksheet}/range(address='{a_addr}')/values",
                json={"values": [[date_str]]}
            )

            # Celular da hora corrente (ex.: às 10h -> coluna "10" = C)
            col = _col_for_hour(hour)
            addr = f"{_col_letter(col)}{row}:{_col_letter(col)}{row}"
            sess.patch(
                f"{base}/workbook/worksheets/{self.worksheet}/range(address='{addr}')/values",
                json={"values": [[int(entradas_count)]]}
            )

            # Atualiza TotalEntradas (O) = SUM(B..N)
            sum_addr = f"O{row}:O{row}"  # TOTAL_COL = 15 -> O
            b_letter = _col_letter(FIRST_HOUR_COL)    # B
            n_letter = _col_letter(LAST_HOUR_COL)     # N
            sess.patch(
                f"{base}/workbook/worksheets/{self.worksheet}/range(address='{sum_addr}')/formula",
                json={"formulas": [[f"=SUM({b_letter}{row}:{n_letter}{row})"]]}
            )
            return True

        # ===================== CSV fallback =====================
        # CSV no formato "date,hour,entradas" + arquivo de cabeçalho mensal
        exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["date", "hour", "entradas"])
            w.writerow([date_str, hour, int(entradas_count)])
        return True