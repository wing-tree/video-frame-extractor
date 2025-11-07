import sys
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from PIL import Image
import numpy as np


class VideoFrameExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_capture = None
        self.current_frame = None
        self.total_frames = 0
        self.fps = 0
        self.video_path = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('비디오 프레임 추출기')
        self.setGeometry(100, 100, 1200, 800)

        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 스크롤 영역 추가
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignCenter)

        # 드래그 앤 드롭 영역 / 비디오 표시 영역
        self.video_label = QLabel('비디오 파일을 여기에 드래그하세요')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                background-color: #f0f0f0;
                font-size: 18px;
                color: #666;
            }
        """)
        self.video_label.setScaledContents(False)

        scroll_area.setWidget(self.video_label)
        layout.addWidget(scroll_area)

        # 타임 정보 레이블
        self.time_label = QLabel('00:00:00.000 / 00:00:00.000')
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.time_label)

        # 타임라인 슬라이더
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.timeline_slider)

        # 컨트롤 버튼들
        control_layout = QHBoxLayout()

        self.open_button = QPushButton('파일 열기')
        self.open_button.clicked.connect(self.open_file)
        control_layout.addWidget(self.open_button)

        self.capture_button = QPushButton('캡처 (WebP 저장)')
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.capture_button)

        layout.addLayout(control_layout)

        # 드래그 앤 드롭 활성화
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            video_file = files[0]
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                self.load_video(video_file)
            else:
                QMessageBox.warning(self, '오류', '지원하는 비디오 파일이 아닙니다.')

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, '비디오 파일 선택', '',
            'Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv)'
        )
        if file_name:
            self.load_video(file_name)

    def load_video(self, video_path):
        if self.video_capture:
            self.video_capture.release()

        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)

        if not self.video_capture.isOpened():
            QMessageBox.critical(self, '오류', '비디오를 열 수 없습니다.')
            return

        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)

        self.timeline_slider.setMaximum(self.total_frames - 1)
        self.timeline_slider.setEnabled(True)
        self.timeline_slider.setValue(0)
        self.capture_button.setEnabled(True)

        self.show_frame(0)

    def show_frame(self, frame_number):
        if not self.video_capture:
            return

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_capture.read()

        if ret:
            self.current_frame = frame

            # OpenCV BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 원본 크기로 표시 (스케일링 없음)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
            self.video_label.resize(pixmap.size())

            # 타임 업데이트
            current_time = frame_number / self.fps if self.fps > 0 else 0
            total_time = self.total_frames / self.fps if self.fps > 0 else 0
            self.time_label.setText(
                f'{self.format_time(current_time)} / {self.format_time(total_time)}'
            )

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}'

    def on_slider_change(self, value):
        self.show_frame(value)

    def capture_frame(self):
        if self.current_frame is None:
            QMessageBox.warning(self, '오류', '캡처할 프레임이 없습니다.')
            return

        # 비디오 파일명 기반으로 기본 이름 생성
        if self.video_path:
            video_filename = Path(self.video_path).stem  # 확장자 제외한 파일명
            default_name = f'{video_filename}.webp'
        else:
            current_frame_num = self.timeline_slider.value()
            default_name = f'frame_{current_frame_num:06d}.webp'

        save_path, _ = QFileDialog.getSaveFileName(
            self, '프레임 저장', default_name, 'WebP Files (*.webp)'
        )

        if save_path:
            try:
                # OpenCV BGR to RGB
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

                # Pillow로 WebP 저장 (알파채널 없음)
                pil_image = Image.fromarray(frame_rgb)
                pil_image = pil_image.convert('RGB')  # 알파채널 제거
                pil_image.save(save_path, 'webp', quality=75)

                QMessageBox.information(self, '성공', f'프레임이 저장되었습니다:\n{save_path}')
            except Exception as e:
                QMessageBox.critical(self, '오류', f'저장 실패:\n{str(e)}')

    def closeEvent(self, event):
        if self.video_capture:
            self.video_capture.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = VideoFrameExtractor()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()