import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QProgressBar,
                             QMessageBox, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import io


class ComparisonThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            # 원본 이미지 로드
            original = cv2.imread(self.image_path)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

            # 원본 파일 크기
            original_size = Path(self.image_path).stat().st_size

            results = []

            # Quality 75부터 100까지 테스트
            for quality in range(75, 101):
                self.progress.emit(quality - 75 + 1)

                # PIL로 WebP 저장 (메모리에)
                pil_image = Image.fromarray(original_rgb)
                pil_image = pil_image.convert('RGB')

                # 메모리에 저장
                buffer = io.BytesIO()
                pil_image.save(buffer, 'webp', quality=quality)
                compressed_size = buffer.tell()

                # 압축된 이미지 다시 읽기
                buffer.seek(0)
                compressed_pil = Image.open(buffer)
                compressed_array = np.array(compressed_pil)
                compressed_bgr = cv2.cvtColor(compressed_array, cv2.COLOR_RGB2BGR)
                compressed_gray = cv2.cvtColor(compressed_bgr, cv2.COLOR_BGR2GRAY)

                # SSIM 계산 (그레이스케일로)
                ssim_score = ssim(original_gray, compressed_gray)

                # 압축률 계산
                compression_ratio = (1 - compressed_size / original_size) * 100

                results.append({
                    'quality': quality,
                    'ssim': ssim_score,
                    'size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'size_mb': compressed_size / (1024 * 1024)
                })

            # Lossless 테스트
            self.progress.emit(27)
            pil_image = Image.fromarray(original_rgb)
            pil_image = pil_image.convert('RGB')
            buffer = io.BytesIO()
            pil_image.save(buffer, 'webp', lossless=True)
            lossless_size = buffer.tell()

            results.append({
                'quality': 'Lossless',
                'ssim': 1.0,  # 무손실은 완벽
                'size': lossless_size,
                'compression_ratio': (1 - lossless_size / original_size) * 100,
                'size_mb': lossless_size / (1024 * 1024)
            })

            self.result.emit(results)

        except Exception as e:
            print(f"Error: {e}")


class WebPQualityCompare(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.comparison_thread = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('WebP Quality & SSIM 비교 도구')
        self.setGeometry(100, 100, 1000, 700)

        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 제목
        title_label = QLabel('WebP Quality 비교 (Quality 75-100 + Lossless)')
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # 파일 정보
        self.file_label = QLabel('이미지를 선택하세요')
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.file_label)

        # 버튼
        button_layout = QHBoxLayout()

        self.select_button = QPushButton('이미지 선택')
        self.select_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_button)

        self.compare_button = QPushButton('비교 시작')
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self.start_comparison)
        self.compare_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.compare_button)

        self.export_button = QPushButton('CSV 내보내기')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        button_layout.addWidget(self.export_button)

        layout.addLayout(button_layout)

        # 진행바
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(27)  # 75-100 (26개) + lossless (1개)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 결과 테이블
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels([
            'Quality', 'SSIM', '파일 크기 (MB)', '압축률 (%)', '원본 대비', '효율성'
        ])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setAlternatingRowColors(True)
        layout.addWidget(self.result_table)

        # 요약 정보
        self.summary_label = QLabel('')
        self.summary_label.setStyleSheet("padding: 10px; background-color: #e8f5e9; border-radius: 5px;")
        layout.addWidget(self.summary_label)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, '이미지 선택', '',
            'Image Files (*.png *.jpg *.jpeg *.bmp *.webp)'
        )
        if file_name:
            self.image_path = file_name
            file_size = Path(file_name).stat().st_size / (1024 * 1024)
            self.file_label.setText(f'선택된 파일: {Path(file_name).name} ({file_size:.2f} MB)')
            self.compare_button.setEnabled(True)
            self.result_table.setRowCount(0)
            self.summary_label.setText('')
            self.export_button.setEnabled(False)

    def start_comparison(self):
        if not self.image_path:
            return

        self.compare_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.result_table.setRowCount(0)

        self.comparison_thread = ComparisonThread(self.image_path)
        self.comparison_thread.progress.connect(self.update_progress)
        self.comparison_thread.result.connect(self.display_results)
        self.comparison_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def display_results(self, results):
        self.result_table.setRowCount(len(results))

        original_size = Path(self.image_path).stat().st_size

        best_ssim_quality = None
        best_ssim_score = 0

        best_efficiency_quality = None
        best_efficiency_score = 0

        for i, result in enumerate(results):
            # 효율성 계산 (SSIM / MB) - 높을수록 좋음
            efficiency = result['ssim'] / result['size_mb'] if result['size_mb'] > 0 else 0
            result['efficiency'] = efficiency

            # Quality
            quality_item = QTableWidgetItem(str(result['quality']))
            quality_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 0, quality_item)

            # SSIM
            ssim_item = QTableWidgetItem(f"{result['ssim']:.6f}")
            ssim_item.setTextAlignment(Qt.AlignCenter)
            if result['ssim'] >= 0.99 and result['quality'] != 'Lossless':
                ssim_item.setBackground(Qt.green)
            self.result_table.setItem(i, 1, ssim_item)

            # 파일 크기
            size_item = QTableWidgetItem(f"{result['size_mb']:.3f}")
            size_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 2, size_item)

            # 압축률
            comp_item = QTableWidgetItem(f"{result['compression_ratio']:.2f}%")
            comp_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 3, comp_item)

            # 원본 대비
            ratio = result['size'] / original_size
            ratio_item = QTableWidgetItem(f"{ratio:.2%}")
            ratio_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 4, ratio_item)

            # 효율성 (SSIM/MB)
            efficiency_item = QTableWidgetItem(f"{efficiency:.3f}")
            efficiency_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 5, efficiency_item)

            # Best SSIM 찾기 (lossless 제외)
            if result['quality'] != 'Lossless' and result['ssim'] > best_ssim_score:
                best_ssim_score = result['ssim']
                best_ssim_quality = result['quality']

            # Best 효율성 찾기 (lossless 제외)
            if result['quality'] != 'Lossless' and efficiency > best_efficiency_score:
                best_efficiency_score = efficiency
                best_efficiency_quality = result['quality']

        # 최고 효율성 하이라이트
        for i, result in enumerate(results):
            if result['quality'] == best_efficiency_quality:
                efficiency_item = self.result_table.item(i, 5)
                efficiency_item.setBackground(Qt.yellow)
                # 전체 행 하이라이트
                for col in range(6):
                    item = self.result_table.item(i, col)
                    if item:
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)

        # 요약
        original_mb = original_size / (1024 * 1024)
        lossless_result = results[-1]
        best_lossy = max([r for r in results if r['quality'] != 'Lossless'], key=lambda x: x['ssim'])
        best_efficiency_result = [r for r in results if r['quality'] == best_efficiency_quality][0]

        summary_text = f"""
        <b>원본 파일:</b> {original_mb:.3f} MB<br>
        <b>Lossless WebP:</b> {lossless_result['size_mb']:.3f} MB ({lossless_result['compression_ratio']:.2f}% 절감, SSIM: 1.0)<br>
        <b>최고 SSIM (Lossy):</b> Quality {best_lossy['quality']}, SSIM {best_lossy['ssim']:.6f}, {best_lossy['size_mb']:.3f} MB<br>
        <b>⭐ 최고 효율 (추천):</b> Quality {best_efficiency_quality}, SSIM {best_efficiency_result['ssim']:.6f}, {best_efficiency_result['size_mb']:.3f} MB (효율: {best_efficiency_score:.3f})<br>
        <b>참고:</b> Quality 95 이상 권장 (SSIM ≥ 0.99)
        """
        self.summary_label.setText(summary_text)

        self.progress_bar.setVisible(False)
        self.compare_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.export_button.setEnabled(True)

        self.results_data = results

    def export_csv(self):
        if not hasattr(self, 'results_data'):
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, 'CSV 저장', 'webp_comparison.csv', 'CSV Files (*.csv)'
        )

        if file_name:
            try:
                with open(file_name, 'w', encoding='utf-8-sig') as f:
                    f.write('Quality,SSIM,Size_MB,Compression_Ratio,Size_Bytes,Efficiency\n')
                    for result in self.results_data:
                        efficiency = result.get('efficiency', 0)
                        f.write(f"{result['quality']},{result['ssim']:.6f},"
                                f"{result['size_mb']:.6f},{result['compression_ratio']:.2f},"
                                f"{result['size']},{efficiency:.6f}\n")

                QMessageBox.information(self, '성공', f'CSV 파일이 저장되었습니다:\n{file_name}')
            except Exception as e:
                QMessageBox.critical(self, '오류', f'저장 실패:\n{str(e)}')


def main():
    app = QApplication(sys.argv)
    window = WebPQualityCompare()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()