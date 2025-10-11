# reporter.py - VERSÃO CORRIGIDA COM LOGS DETALHADOS
import os
import csv
import json
import datetime as dt
from typing import Optional, Any, Dict, List
import requests
import msal

# Opcional (Google Sheets)
try:
    import gspread  # type: ignore
    from google.oauth2 import service_account  # type: ignore
    GSHEETS_AVAILABLE = True
except ImportError as e:
    gspread = None  # type: ignore
    service_account = None  # type: ignore
    GSHEETS_AVAILABLE = False
    print(f"⚠ Google Sheets não disponível: {e}")


# Fuso fixo GMT-3 (sem DST)
def _now_local() -> dt.datetime:
    """Retorna datetime atual em GMT-3 (sem DST)"""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=3)


def _col_letter(n: int) -> str:
    """Converte índice numérico para letra de coluna Excel (1=A, 2=B, ...)"""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ============ CONFIGURAÇÃO DA JANELA DE HORÁRIOS ============
# IMPORTANTE: Janela ajustada para 9h-21h (13 colunas)
# Colunas: A=Data | B..M=9h..21h (13 cols) | N=TotalEntradas
DATA_COL = 1
FIRST_HOUR = 9
LAST_HOUR = 21
FIRST_HOUR_COL = DATA_COL + 1          # 2 -> B
LAST_HOUR_COL = FIRST_HOUR_COL + (LAST_HOUR - FIRST_HOUR)  # 14 -> M (13 colunas)
TOTAL_COL = LAST_HOUR_COL + 1          # 15 -> N


def _row_for_day(day: int) -> int:
    """Linha para um dia do mês (dia 1 = linha 2, pois linha 1 é header)"""
    return 1 + day


def _col_for_hour(hour: int) -> int:
    """Retorna índice da coluna para uma hora específica"""
    if hour < FIRST_HOUR or hour > LAST_HOUR:
        raise ValueError(f"Hora {hour} fora da janela [{FIRST_HOUR}..{LAST_HOUR}]")
    return FIRST_HOUR_COL + (hour - FIRST_HOUR)


def _headers() -> List[str]:
    """Gera cabeçalhos: Data | 10 | 11 | ... | 21 | TotalEntradas"""
    return ["Data"] + [f"{h:02d}" for h in range(FIRST_HOUR, LAST_HOUR + 1)] + ["TotalEntradas"]


class Reporter:
    def __init__(self) -> None:
        self.mode = os.getenv("REPORT_MODE", "gsheets_wide")
        print(f"📊 Reporter inicializado: mode={self.mode}")
        
        # Credenciais Excel (Graph API)
        self.tenant_id = os.getenv("EXCEL_TENANT_ID")
        self.client_id = os.getenv("EXCEL_CLIENT_ID")
        self.client_sec = os.getenv("EXCEL_CLIENT_SECRET")
        self.drive_id = os.getenv("EXCEL_DRIVE_ID")
        self.item_id = os.getenv("EXCEL_ITEM_ID")
        self.worksheet = os.getenv("EXCEL_WORKSHEET", "Planilha1")
        
        # CSV fallback
        self.csv_path = os.getenv("CSV_PATH", "movimento_entradas.csv")
        
        # Google Sheets
        self.gs_json = os.getenv("GSHEETS_CREDENTIALS_JSON")
        self.gs_sheet = os.getenv("GSHEETS_SPREADSHEET_ID")
        self.gs_ws_name = os.getenv("GSHEETS_WORKSHEET", "Entradas_por_hora")
        
        # Valida configuração
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Valida se as credenciais necessárias estão disponíveis"""
        if self.mode == "graph_excel":
            missing = []
            for var in ["EXCEL_TENANT_ID", "EXCEL_CLIENT_ID", "EXCEL_CLIENT_SECRET", 
                       "EXCEL_DRIVE_ID", "EXCEL_ITEM_ID"]:
                if not os.getenv(var):
                    missing.append(var)
            if missing:
                print(f"⚠ Credenciais Excel faltando: {', '.join(missing)}")
        
        elif self.mode == "gsheets_wide":
            if not GSHEETS_AVAILABLE:
                print("✗ Modo gsheets_wide selecionado mas bibliotecas não disponíveis!")
                return
            
            if not self.gs_json:
                print("⚠ GSHEETS_CREDENTIALS_JSON não configurado")
            if not self.gs_sheet:
                print("⚠ GSHEETS_SPREADSHEET_ID não configurado")
            else:
                print(f"✓ Google Sheets: spreadsheet_id={self.gs_sheet[:20]}...")

    # ===================== Excel (Graph API) =====================
    def _get_graph_session(self) -> requests.Session:
        """Autentica e retorna sessão com token do Graph API"""
        print("🔐 Autenticando no Microsoft Graph...")
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=authority,
            client_credential=self.client_sec
        )
        scopes = ["https://graph.microsoft.com/.default"]
        token = app.acquire_token_for_client(scopes=scopes)
        
        if "access_token" not in token:
            error_msg = token.get('error_description', 'Unknown error')
            raise RuntimeError(f"Falha autenticação MSAL: {error_msg}")
        
        print("✓ Token obtido com sucesso")
        sess = requests.Session()
        sess.headers.update({"Authorization": f"Bearer {token['access_token']}"})
        return sess

    def _graph_base_url(self) -> str:
        """URL base para operações no Excel via Graph"""
        return f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/items/{self.item_id}"

    def _ensure_month_header(self, session: requests.Session, base_url: str) -> None:
        """Garante que a linha 1 tem o cabeçalho correto"""
        headers = _headers()
        last_col_letter = _col_letter(TOTAL_COL)
        url = f"{base_url}/workbook/worksheets/{self.worksheet}/range(address='A1:{last_col_letter}1')/values"
        
        try:
            response = session.patch(url, json={"values": [headers]})
            response.raise_for_status()
            print(f"✓ Cabeçalho atualizado: A1:{last_col_letter}1")
        except Exception as e:
            print(f"⚠ Erro ao atualizar cabeçalho: {e}")

    # ===================== Google Sheets =====================
    def _write_to_gsheets(self, entradas_count: int, when: dt.datetime) -> bool:
        """Escreve dados no Google Sheets (modo wide: 10h-21h)"""
        if not GSHEETS_AVAILABLE:
            raise RuntimeError("Bibliotecas Google Sheets não disponíveis")
        
        day = when.day
        hour = when.hour
        date_str = when.strftime("%Y-%m-%d")
        
        print(f"📝 Google Sheets: gravando {entradas_count} entradas para {date_str} {hour:02d}h")
        
        # Autentica
        try:
            info = json.loads(self.gs_json)  # type: ignore
            print(f"✓ Credenciais JSON parseadas (type: {info.get('type', 'unknown')})")
        except Exception as e:
            raise RuntimeError(f"GSHEETS_CREDENTIALS_JSON inválido: {e}")
        
        creds = service_account.Credentials.from_service_account_info(  # type: ignore
            info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)  # type: ignore
        print("✓ Cliente Google Sheets autorizado")
        
        # Abre planilha
        try:
            sh = client.open_by_key(self.gs_sheet)
            print(f"✓ Planilha aberta: {sh.title}")
        except Exception as e:
            raise RuntimeError(f"Erro ao abrir spreadsheet {self.gs_sheet}: {e}")
        
        # Abre/cria worksheet
        try:
            ws = sh.worksheet(self.gs_ws_name)
            print(f"✓ Worksheet encontrada: {self.gs_ws_name}")
        except gspread.WorksheetNotFound:  # type: ignore
            print(f"⚠ Worksheet {self.gs_ws_name} não existe - criando...")
            ws = sh.add_worksheet(title=self.gs_ws_name, rows=200, cols=15)
            print("✓ Worksheet criada")
        
        # Garante cabeçalho (A1:N1)
        headers = _headers()
        try:
            ws.update('A1:N1', [headers], value_input_option='USER_ENTERED')
            print(f"✓ Cabeçalho atualizado: {headers}")
        except Exception as e:
            print(f"⚠ Erro ao atualizar cabeçalho: {e}")
        
        # Localiza ou cria linha da data
        try:
            col_a = ws.col_values(1)
            row = None
            for idx, val in enumerate(col_a, start=1):
                if idx == 1:  # header
                    continue
                if val == date_str:
                    row = idx
                    break
            
            if row is None:
                # Adiciona nova linha
                skeleton = [date_str] + ["" for _ in range(12)] + [""]
                ws.append_row(skeleton, value_input_option="USER_ENTERED")
                row = len(col_a) + 1
                print(f"✓ Nova linha criada para {date_str}: linha {row}")
            else:
                print(f"✓ Linha existente encontrada para {date_str}: linha {row}")
        except Exception as e:
            raise RuntimeError(f"Erro ao localizar/criar linha: {e}")
        
        # Calcula coluna da hora (10h=B=2, 21h=M=13)
        col_idx = _col_for_hour(hour)
        col_letter = _col_letter(col_idx)
        cell_addr = f"{col_letter}{row}"
        
        print(f"📍 Gravando {entradas_count} em {cell_addr} (hora {hour}h)")
        
        # Atualiza célula da hora
        try:
            ws.update(cell_addr, [[int(entradas_count)]], value_input_option='USER_ENTERED')
            print(f"✓ Valor gravado: {cell_addr} = {entradas_count}")
        except Exception as e:
            raise RuntimeError(f"Erro ao gravar valor: {e}")
        
        # Atualiza fórmula de total (coluna N)
        total_cell = f"N{row}"
        total_formula = f"=SUM(B{row}:M{row})"
        
        try:
            ws.update(total_cell, [[total_formula]], value_input_option='USER_ENTERED')
            print(f"✓ Fórmula total atualizada: {total_cell} = {total_formula}")
        except Exception as e:
            print(f"⚠ Erro ao atualizar fórmula total: {e}")
        
        return True

    # ===================== MÉTODO PRINCIPAL =====================
    def write_hourly(self, entradas_count: int, when: Optional[dt.datetime] = None) -> bool:
        """
        Grava contagem de entradas para uma hora específica.
        
        Args:
            entradas_count: Número de entradas detectadas
            when: Timestamp (usa horário atual GMT-3 se None)
        
        Returns:
            True se gravou com sucesso, False caso contrário
        """
        if when is None:
            when = _now_local()
        
        day = when.day
        hour = when.hour
        date_str = when.strftime("%Y-%m-%d")
        
        print(f"\n{'='*70}")
        print("📊 GRAVANDO DADOS")
        print(f"{'='*70}")
        print(f"Data/hora: {date_str} {hour:02d}:00 (GMT-3)")
        print(f"Entradas: {entradas_count}")
        print(f"Modo: {self.mode}")
        print(f"Janela válida: {FIRST_HOUR}h-{LAST_HOUR}h")
        
        # ✅ CORREÇÃO: Valida janela de horários, mas LOGA o motivo
        if hour < FIRST_HOUR or hour > LAST_HOUR:
            print(f"⚠ AVISO: Hora {hour}h está FORA da janela [{FIRST_HOUR}h-{LAST_HOUR}h]")
            print("   Os dados NÃO serão gravados na planilha")
            print("   Para gravar fora deste horário, ajuste FIRST_HOUR/LAST_HOUR no reporter.py")
            print(f"{'='*70}\n")
            return False
        
        print(f"✓ Hora {hour}h está dentro da janela válida")
        
        try:
            # Excel via Graph API
            if self.mode == "graph_excel":
                sess = self._get_graph_session()
                base = self._graph_base_url()
                
                self._ensure_month_header(sess, base)
                row = _row_for_day(day)
                
                # Atualiza Data (coluna A)
                a_addr = f"A{row}"
                url = f"{base}/workbook/worksheets/{self.worksheet}/range(address='{a_addr}')/values"
                response = sess.patch(url, json={"values": [[date_str]]})
                response.raise_for_status()
                print(f"✓ Data gravada: {a_addr} = {date_str}")
                
                # Atualiza hora específica
                col = _col_for_hour(hour)
                col_letter = _col_letter(col)
                addr = f"{col_letter}{row}"
                url = f"{base}/workbook/worksheets/{self.worksheet}/range(address='{addr}')/values"
                response = sess.patch(url, json={"values": [[int(entradas_count)]]})
                response.raise_for_status()
                print(f"✓ Valor gravado: {addr} = {entradas_count}")
                
                # Atualiza fórmula total
                sum_addr = f"{_col_letter(TOTAL_COL)}{row}"
                b_letter = _col_letter(FIRST_HOUR_COL)
                m_letter = _col_letter(LAST_HOUR_COL)
                url = f"{base}/workbook/worksheets/{self.worksheet}/range(address='{sum_addr}')/formula"
                response = sess.patch(url, json={"formulas": [[f"=SUM({b_letter}{row}:{m_letter}{row})"]]})
                response.raise_for_status()
                print(f"✓ Fórmula total gravada: {sum_addr}")
                
                print(f"{'='*70}")
                print("✅ DADOS GRAVADOS COM SUCESSO (Excel)")
                print(f"{'='*70}\n")
                return True
            
            # Google Sheets
            elif self.mode == "gsheets_wide":
                result = self._write_to_gsheets(entradas_count, when)
                print(f"{'='*70}")
                print("✅ DADOS GRAVADOS COM SUCESSO (Google Sheets)")
                print(f"{'='*70}\n")
                return result
            
            # CSV fallback
            else:
                exists = os.path.exists(self.csv_path)
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if not exists:
                        w.writerow(["date", "hour", "entradas"])
                    w.writerow([date_str, hour, int(entradas_count)])
                
                print(f"✓ Dados gravados no CSV: {self.csv_path}")
                print(f"{'='*70}")
                print("✅ DADOS GRAVADOS COM SUCESSO (CSV)")
                print(f"{'='*70}\n")
                return True
        
        except Exception as e:
            print(f"\n{'='*70}")
            print("❌ ERRO AO GRAVAR DADOS")
            print(f"{'='*70}")
            print(f"Erro: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*70}\n")
            return False