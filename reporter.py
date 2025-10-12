# reporter.py - FORMATO LONG (Data | Hora | Quantidade)
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


# ============ CONFIGURAÇÃO (mantida para compatibilidade) ============
FIRST_HOUR = 9
LAST_HOUR = 20


def _headers() -> List[str]:
    """Gera cabeçalhos para formato long: Data | Hora | Quantidade"""
    return ["Data", "Hora", "Quantidade"]


class Reporter:
    def __init__(self) -> None:
        self.mode = os.getenv("REPORT_MODE", "gsheets")
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
        
        elif self.mode == "gsheets":
            if not GSHEETS_AVAILABLE:
                print("✗ Modo gsheets selecionado mas bibliotecas não disponíveis!")
                return
            
            if not self.gs_json:
                print("⚠ GSHEETS_CREDENTIALS_JSON não configurado")
            if not self.gs_sheet:
                print("⚠ GSHEETS_SPREADSHEET_ID não configurado")
            else:
                print(f"✓ Google Sheets: spreadsheet_id={self.gs_sheet[:20]}...")

    # ===================== Google Sheets (FORMATO LONG) =====================
    def _write_to_gsheets(self, entradas_count: int, when: dt.datetime) -> bool:
        """Escreve dados no Google Sheets (modo long: Data | Hora | Quantidade)"""
        if not GSHEETS_AVAILABLE:
            raise RuntimeError("Bibliotecas Google Sheets não disponíveis")
        
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
            ws = sh.add_worksheet(title=self.gs_ws_name, rows=1000, cols=3)
            print("✓ Worksheet criada")
        
        # Garante cabeçalho (A1:C1)
        headers = _headers()
        try:
            current_header = ws.row_values(1)
            if current_header != headers:
                ws.update('A1:C1', [headers], value_input_option='USER_ENTERED')
                print(f"✓ Cabeçalho atualizado: {headers}")
        except Exception as e:
            print(f"⚠ Erro ao verificar cabeçalho: {e}")
            ws.update('A1:C1', [headers], value_input_option='USER_ENTERED')
        
        # Verifica se já existe uma linha para esta data+hora
        try:
            all_values = ws.get_all_values()
            row_to_update = None
            
            for idx, row in enumerate(all_values, start=1):
                if idx == 1:  # pula cabeçalho
                    continue
                if len(row) >= 2 and row[0] == date_str and row[1] == str(hour):
                    row_to_update = idx
                    print(f"✓ Linha existente encontrada: linha {row_to_update}")
                    break
            
            if row_to_update:
                # Atualiza linha existente
                ws.update(f'C{row_to_update}', [[int(entradas_count)]], value_input_option='USER_ENTERED')
                print(f"✓ Valor atualizado: C{row_to_update} = {entradas_count}")
            else:
                # Adiciona nova linha
                new_row = [date_str, hour, int(entradas_count)]
                ws.append_row(new_row, value_input_option="USER_ENTERED")
                print(f"✓ Nova linha adicionada: {new_row}")
        
        except Exception as e:
            raise RuntimeError(f"Erro ao gravar dados: {e}")
        
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
        print(f"✓ Gravando para hora {hour}h")
        
        try:
            # Google Sheets
            if self.mode == "gsheets":
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