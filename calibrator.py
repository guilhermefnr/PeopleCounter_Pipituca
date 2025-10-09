import cv2
import json
import os
import time
import numpy as np
from datetime import datetime

# Configurações
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "line_config.json")
UI_W, UI_H = 640, 480
RTSP_URL = "rtsp://admin:111229@192.168.1.3:554/cam/realmonitor?channel=1&subtype=1"

# Otimizações RTSP
FFMPEG_OPTS = (
    "rtsp_transport;tcp|stimeout;5000000|max_delay;0|"
    "buffer_size;512000|rtsp_flags;prefer_tcp|fflags;nobuffer"
)

class LineCalibrator:
    def __init__(self):
        self.line_start = None
        self.line_end = None
        self.preview_mode = False
        self.show_grid = True
        self.zoom_factor = 1.0
        self.offset_x, self.offset_y = 0, 0
        
        # Configurações visuais
        self.line_thickness = 3
        self.point_size = 8
        self.gate_width = 60
        
        # Estados de calibração
        self.calibration_complete = False
        self.last_save_time = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback aprimorado para mouse com zoom e precisão"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ajusta coordenadas para zoom
            real_x = int(x / self.zoom_factor + self.offset_x)
            real_y = int(y / self.zoom_factor + self.offset_y)
            
            # Garante que as coordenadas estão dentro dos limites
            real_x = max(0, min(UI_W - 1, real_x))
            real_y = max(0, min(UI_H - 1, real_y))
            
            if self.line_start is None:
                self.line_start = (real_x, real_y)
                self.line_end = None
                print(f"📍 INÍCIO marcado: {self.line_start}")
                
            elif self.line_end is None:
                self.line_end = (real_x, real_y)
                self.calibration_complete = True
                print(f"🎯 FIM marcado: {self.line_end}")
                print("✅ Linha completa! Pressione [S] para salvar")
                
            else:
                # Reset - nova linha
                self.line_start = (real_x, real_y)
                self.line_end = None
                self.calibration_complete = False
                print(f"🔄 RESET - Nova linha: {self.line_start}")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Botão direito para limpar
            self.clear_calibration()

    def clear_calibration(self):
        """Limpa calibração atual"""
        self.line_start = None
        self.line_end = None
        self.calibration_complete = False
        print("🧹 Calibração limpa")

    def get_line_info(self):
        """Calcula informações detalhadas da linha"""
        if not (self.line_start and self.line_end):
            return None
        
        dx = abs(self.line_end[0] - self.line_start[0])
        dy = abs(self.line_end[1] - self.line_start[1])
        length = int(np.sqrt(dx**2 + dy**2))
        
        # Determina orientação com maior precisão
        angle = np.degrees(np.arctan2(dy, dx))
        
        if dx < dy:
            line_type = "vertical"
            line_position = (self.line_start[0] + self.line_end[0]) // 2
            primary_axis = "y"
        else:
            line_type = "horizontal" 
            line_position = (self.line_start[1] + self.line_end[1]) // 2
            primary_axis = "x"
        
        return {
            'type': line_type,
            'position': line_position,
            'length': length,
            'angle': angle,
            'primary_axis': primary_axis,
            'dx': dx,
            'dy': dy
        }

    def save_configuration(self):
        """Salva configuração com validação completa"""
        if not self.calibration_complete:
            print("❌ Complete a calibração antes de salvar!")
            return False
        
        line_info = self.get_line_info()
        if not line_info:
            print("❌ Informações da linha inválidas!")
            return False
        
        # Validações
        if line_info['length'] < 30:
            print("❌ Linha muito curta! Mínimo 30 pixels")
            return False
        
        if line_info['length'] > min(UI_W, UI_H) * 0.8:
            print("⚠️ Linha muito longa, pode causar problemas de detecção")
        
        # Configuração completa
        config = {
            # Coordenadas da linha
            "line_start": list(self.line_start),
            "line_end": list(self.line_end),
            
            # Informações calculadas
            "line_type": line_info['type'],
            "line_position": line_info['position'],
            "line_length": line_info['length'],
            "line_angle": line_info['angle'],
            
            # Configurações de detecção
            "gate_width": self.gate_width,
            "entry_side": "left" if line_info['type'] == "vertical" else "top",
            
            # Metadados do frame
            "frame_width": UI_W,
            "frame_height": UI_H,
            "calibration_timestamp": time.time(),
            "calibration_date": datetime.now().isoformat(),
            
            # Versão e validação
            "config_version": "2.0",
            "calibrator": "advanced_yolo_counter",
            
            # Configurações adicionais para tracking
            "detection_area": {
                "x": max(0, line_info['position'] - self.gate_width * 2) if line_info['type'] == "vertical" else 0,
                "y": max(0, line_info['position'] - self.gate_width * 2) if line_info['type'] == "horizontal" else 0,
                "width": self.gate_width * 4 if line_info['type'] == "vertical" else UI_W,
                "height": self.gate_width * 4 if line_info['type'] == "horizontal" else UI_H
            }
        }
        
        try:
            # Backup da configuração anterior se existir
            if os.path.exists(OUTPUT_FILE):
                backup_file = OUTPUT_FILE.replace('.json', f'_backup_{int(time.time())}.json')
                import shutil
                shutil.copy2(OUTPUT_FILE, backup_file)
                print(f"💾 Backup salvo: {backup_file}")
            
            # Salva nova configuração
            with open(OUTPUT_FILE, "w") as f:
                json.dump(config, f, indent=2)
            
            self.last_save_time = time.time()
            
            print(f"✅ Configuração salva: {OUTPUT_FILE}")
            print(f"   📏 Linha: {self.line_start} → {self.line_end}")
            print(f"   📐 Tipo: {line_info['type']} ({line_info['length']}px)")
            print(f"   🎯 Ângulo: {line_info['angle']:.1f}°")
            print(f"   🚪 Portal: {self.gate_width}px")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar: {e}")
            return False

    def load_existing_configuration(self):
        """Carrega configuração existente com validação"""
        if not os.path.exists(OUTPUT_FILE):
            print("📂 Nenhuma configuração anterior encontrada")
            return False
        
        try:
            with open(OUTPUT_FILE, "r") as f:
                config = json.load(f)
            
            # Valida estrutura
            required_fields = ["line_start", "line_end"]
            if not all(field in config for field in required_fields):
                print("⚠️ Configuração anterior inválida")
                return False
            
            self.line_start = tuple(config["line_start"])
            self.line_end = tuple(config["line_end"])
            self.calibration_complete = True
            
            # Carrega configurações adicionais se disponíveis
            if "gate_width" in config:
                self.gate_width = config["gate_width"]
            
            line_info = self.get_line_info()
            
            print(f"📂 Configuração carregada:")
            print(f"   📏 Linha: {self.line_start} → {self.line_end}")
            print(f"   📐 Tipo: {line_info['type']} ({line_info['length']}px)")
            print(f"   📅 Salva: {config.get('calibration_date', 'data desconhecida')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar configuração: {e}")
            return False

    def draw_interface(self, frame):
        """Desenha interface visual completa"""
        display_frame = frame.copy()
        
        # Grid de auxílio (opcional)
        if self.show_grid:
            self.draw_grid(display_frame)
        
        # Linha principal
        if self.line_start:
            # Ponto inicial
            cv2.circle(display_frame, self.line_start, self.point_size, (0, 255, 255), -1)
            cv2.circle(display_frame, self.line_start, self.point_size + 2, (0, 0, 0), 2)
            
            if self.line_end:
                # Ponto final
                cv2.circle(display_frame, self.line_end, self.point_size, (0, 255, 0), -1)
                cv2.circle(display_frame, self.line_end, self.point_size + 2, (0, 0, 0), 2)
                
                # Linha conectando
                cv2.line(display_frame, self.line_start, self.line_end, 
                        (0, 255, 255), self.line_thickness + 2)
                cv2.line(display_frame, self.line_start, self.line_end, 
                        (0, 255, 0), self.line_thickness)
                
                # Área do portal
                self.draw_gate_area(display_frame)
                
                # Informações da linha
                line_info = self.get_line_info()
                if line_info:
                    self.draw_line_info(display_frame, line_info)
        
        # Interface de status
        self.draw_status_panel(display_frame)
        
        # Instruções
        self.draw_instructions(display_frame)
        
        return display_frame

    def draw_grid(self, frame):
        """Desenha grid de auxílio"""
        grid_spacing = 40
        grid_color = (50, 50, 50)
        
        # Linhas verticais
        for x in range(0, UI_W, grid_spacing):
            cv2.line(frame, (x, 0), (x, UI_H), grid_color, 1)
        
        # Linhas horizontais  
        for y in range(0, UI_H, grid_spacing):
            cv2.line(frame, (0, y), (UI_W, y), grid_color, 1)

    def draw_gate_area(self, frame):
        """Desenha área do portal de detecção"""
        line_info = self.get_line_info()
        if not line_info:
            return
        
        gate_color = (100, 100, 255)
        
        if line_info['type'] == "vertical":
            gate_left = max(0, line_info['position'] - self.gate_width)
            gate_right = min(UI_W, line_info['position'] + self.gate_width)
            
            # Área do portal
            cv2.rectangle(frame, (gate_left, 0), (gate_right, UI_H), gate_color, 2)
            
            # Linhas de referência
            cv2.line(frame, (gate_left, 0), (gate_left, UI_H), gate_color, 1)
            cv2.line(frame, (gate_right, 0), (gate_right, UI_H), gate_color, 1)
            
        else:  # horizontal
            gate_top = max(0, line_info['position'] - self.gate_width)
            gate_bottom = min(UI_H, line_info['position'] + self.gate_width)
            
            # Área do portal
            cv2.rectangle(frame, (0, gate_top), (UI_W, gate_bottom), gate_color, 2)
            
            # Linhas de referência
            cv2.line(frame, (0, gate_top), (UI_W, gate_top), gate_color, 1)
            cv2.line(frame, (0, gate_bottom), (UI_W, gate_bottom), gate_color, 1)

    def draw_line_info(self, frame, line_info):
        """Desenha informações técnicas da linha"""
        info_x, info_y = 10, 100
        line_height = 20
        
        info_bg = (0, 0, 0)
        info_color = (255, 255, 255)
        
        infos = [
            f"Tipo: {line_info['type'].upper()}",
            f"Comprimento: {line_info['length']}px",
            f"Ângulo: {line_info['angle']:.1f}°",
            f"Portal: ±{self.gate_width}px"
        ]
        
        # Fundo
        panel_height = len(infos) * line_height + 10
        cv2.rectangle(frame, (info_x - 5, info_y - 15), 
                     (info_x + 200, info_y + panel_height), info_bg, -1)
        
        # Textos
        for i, info in enumerate(infos):
            y_pos = info_y + i * line_height
            cv2.putText(frame, info, (info_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)

    def draw_status_panel(self, frame):
        """Desenha painel de status principal"""
        panel_height = 60
        
        # Fundo do painel
        cv2.rectangle(frame, (5, 5), (UI_W - 5, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (UI_W - 5, panel_height), (100, 100, 100), 2)
        
        # Status principal
        if self.calibration_complete:
            status_text = "✅ CALIBRAÇÃO COMPLETA - Pressione [S] para salvar"
            status_color = (0, 255, 0)
        elif self.line_start:
            status_text = "📍 Clique no ponto FINAL da linha"
            status_color = (0, 255, 255)
        else:
            status_text = "👆 Clique no ponto INICIAL da linha"
            status_color = (255, 255, 255)
        
        cv2.putText(frame, status_text, (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Informação de salvamento recente
        if self.last_save_time > 0:
            time_since_save = int(time.time() - self.last_save_time)
            save_info = f"Último salvamento: {time_since_save}s atrás"
            cv2.putText(frame, save_info, (15, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    def draw_instructions(self, frame):
        """Desenha instruções de uso"""
        instructions = [
            "S:salvar | L:carregar | C:limpar | G:grid ON/OFF",
            "[ ] :ajustar portal | +/-:espessura | Q:sair"
        ]
        
        y_start = UI_H - 40
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, y_start + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def open_camera(self):
        """Abre conexão otimizada com a câmera"""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_OPTS
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)
            except:
                pass
        
        return cap

    def run(self):
        """Loop principal do calibrador"""
        print("=" * 70)
        print("🎯 CALIBRADOR AVANÇADO - Contador de Pessoas YOLO")
        print("=" * 70)
        print("INSTRUÇÕES DETALHADAS:")
        print("• Clique esquerdo: marcar pontos da linha (início → fim)")
        print("• Clique direito: limpar calibração atual")
        print("• [S]: salvar configuração")
        print("• [L]: carregar configuração existente")
        print("• [C]: limpar linha atual")
        print("• [G]: mostrar/ocultar grid de auxílio")
        print("• [[/]]: ajustar largura do portal de detecção")
        print("• [+/-]: ajustar espessura visual da linha")
        print("• [Q]: sair do calibrador")
        print("-" * 70)
        
        # Tenta carregar configuração existente
        self.load_existing_configuration()
        
        # Conecta à câmera
        cap = self.open_camera()
        if not cap.isOpened():
            print(f"❌ Falha ao conectar à câmera: {RTSP_URL}")
            return
        
        print("✅ Câmera conectada com sucesso")
        
        # Configura janela
        cv2.namedWindow("Calibrador Avançado", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibrador Avançado", UI_W, UI_H)
        cv2.setMouseCallback("Calibrador Avançado", self.mouse_callback)
        
        frame_skip = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_skip += 1
                if frame_skip % 30 == 0:
                    print("⚠️ Problemas de conexão com câmera")
                continue
            
            frame_skip = 0
            frame = cv2.resize(frame, (UI_W, UI_H))
            
            # Aplica interface visual
            display_frame = self.draw_interface(frame)
            
            cv2.imshow("Calibrador Avançado", display_frame)
            
            # Controle de teclado
            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord('q'), 27):  # Q ou ESC
                break
            elif key == ord('s'):  # Salvar
                if self.save_configuration():
                    print("🎉 Configuração salva! Execute o contador agora.")
                else:
                    print("⚠️ Falha ao salvar configuração")
            elif key == ord('l'):  # Load
                if self.load_existing_configuration():
                    print("✅ Configuração carregada")
                else:
                    print("❌ Nenhuma configuração válida encontrada")
            elif key == ord('c'):  # Clear
                self.clear_calibration()
            elif key == ord('g'):  # Grid
                self.show_grid = not self.show_grid
                print(f"Grid: {'ATIVO' if self.show_grid else 'INATIVO'}")
            elif key == ord(']'):  # Aumentar portal
                self.gate_width = min(150, self.gate_width + 10)
                print(f"Largura do portal: {self.gate_width}px")
            elif key == ord('['):  # Diminuir portal
                self.gate_width = max(20, self.gate_width - 10)
                print(f"Largura do portal: {self.gate_width}px")
            elif key == ord('+'):  # Aumentar espessura
                self.line_thickness = min(8, self.line_thickness + 1)
                print(f"Espessura da linha: {self.line_thickness}px")
            elif key == ord('-'):  # Diminuir espessura
                self.line_thickness = max(1, self.line_thickness - 1)
                print(f"Espessura da linha: {self.line_thickness}px")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Calibrador finalizado")

def main():
    try:
        calibrator = LineCalibrator()
        calibrator.run()
    except KeyboardInterrupt:
        print("\n👋 Calibrador interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro no calibrador: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()