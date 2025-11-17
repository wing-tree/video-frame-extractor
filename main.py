import sys
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QMessageBox, QScrollArea, QTextEdit,
                             QSplitter, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent, QFont
from PIL import Image
import numpy as np
import subprocess
import json


class VideoFrameExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_capture = None
        self.current_frame = None
        self.total_frames = 0
        self.fps = 0
        self.video_path = None
        self.last_frame_number = -1
        self.frame_info = []  # 프레임 정보 리스트 (타입, 크기, QP)
        self.avg_sizes = {}  # 타입별 평균 크기

        self.init_ui()
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        self.setWindowTitle('비디오 프레임 추출기')
        self.setGeometry(100, 100, 1400, 800)

        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 스플리터로 좌우 분할
        splitter = QSplitter(Qt.Horizontal)

        # 왼쪽: 비디오 영역
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        # 스크롤 영역 추가
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
        self.stats_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #444;
                padding: 5px;
                color: #e0e0e0;
            }
            QListWidget::item {
                padding: 3px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:hover {
                background-color: #2d2d2d;
            }
            QListWidget::item:selected {
                background-color: #0d47a1;
                color: white;
            }
        """)
        control_layout.addWidget(self.capture_button)

        layout.addLayout(control_layout)

        # 오른쪽: 통계 영역
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        stats_label = QLabel('프레임 품질 순위 (Top 10) - 클릭하면 이동')
        stats_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        right_layout.addWidget(stats_label)

        # QTextEdit 대신 QListWidget 사용 (클릭 가능)
        self.stats_list = QListWidget()
        self.stats_list.setMinimumWidth(350)
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        font.setPointSize(10)
        self.stats_list.setFont(font)
        self.stats_list.setStyleSheet("""
            QListWidget {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                padding: 5px;
            }
            QListWidget::item {
                padding: 3px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        self.stats_list.itemClicked.connect(self.on_stats_item_clicked)
        right_layout.addWidget(self.stats_list)

        # 스플리터에 추가
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)  # 왼쪽이 더 넓게
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

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

    def analyze_frame_quality(self, video_path):
        """비디오의 모든 프레임 타입, 크기, QP 분석"""
        cmd = [
            'ffprobe',
            '-select_streams', 'v:0',
            '-show_frames',
            '-show_entries', 'frame=pict_type,pkt_size,quality',
            '-of', 'json',
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            frame_info = []
            has_quality = False

            for frame in data['frames']:
                frame_type = frame.get('pict_type', '?')
                frame_size = int(frame.get('pkt_size', 0))
                quality = frame.get('quality')

                # quality 값이 있는지 확인
                if quality is not None and not has_quality:
                    has_quality = True

                info = {
                    'type': frame_type,
                    'size': frame_size,
                    'quality': quality
                }
                frame_info.append(info)

            # 통계 출력
            i_count = sum(1 for f in frame_info if f['type'] == 'I')
            p_count = sum(1 for f in frame_info if f['type'] == 'P')
            b_count = sum(1 for f in frame_info if f['type'] == 'B')

            print(f"[INFO] 프레임 분석 완료: I={i_count}, P={p_count}, B={b_count}")

            if has_quality:
                print(f"[INFO] QP 값 지원됨")
            else:
                print(f"[INFO] QP 값 미지원 (크기만 사용)")

            # 타입별 평균 크기 계산
            sizes_by_type = {'I': [], 'P': [], 'B': []}
            for info in frame_info:
                ftype = info['type']
                if ftype in sizes_by_type:
                    sizes_by_type[ftype].append(info['size'])

            avg_sizes = {}
            for ftype, sizes in sizes_by_type.items():
                if sizes:
                    avg_sizes[ftype] = sum(sizes) / len(sizes)

            if avg_sizes:
                print(f"[INFO] 평균 크기 - I: {avg_sizes.get('I', 0):.0f}B, "
                      f"P: {avg_sizes.get('P', 0):.0f}B, "
                      f"B: {avg_sizes.get('B', 0):.0f}B")

            return frame_info, avg_sizes

        except Exception as e:
            print(f"[ERROR] ffprobe 실패: {e}")
            print("[INFO] ffprobe가 설치되어 있지 않거나 실행할 수 없습니다.")
            return [], {}

    def format_time_short(self, frame_number):
        """프레임 번호를 시간으로 변환 (짧은 형식)"""
        if self.fps == 0:
            return "00:00.000"

        seconds = frame_number / self.fps
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f'{minutes:02d}:{secs:06.3f}'

    def on_stats_item_clicked(self, item):
        """통계 항목 클릭 시 해당 프레임으로 이동"""
        # item의 data에 프레임 번호 저장되어 있음
        frame_number = item.data(Qt.UserRole)

        if frame_number is not None:
            # 슬라이더 값으로 변환 (10ms 단위)
            time_seconds = frame_number / self.fps if self.fps > 0 else 0
            slider_value = int(time_seconds * 100)

            print(f"[INFO] 프레임 {frame_number}로 이동 ({self.format_time_short(frame_number)})")
            self.timeline_slider.setValue(slider_value)

    def update_stats_display(self):
        """프레임 품질 통계 표시"""
        self.stats_list.clear()

        if not self.frame_info:
            item = QListWidgetItem("프레임 분석 데이터가 없습니다.")
            self.stats_list.addItem(item)
            return

        # 타입별로 분류하고 크기 순으로 정렬
        frames_by_type = {'I': [], 'P': [], 'B': []}

        for idx, info in enumerate(self.frame_info):
            frame_type = info['type']
            if frame_type in frames_by_type:
                frames_by_type[frame_type].append({
                    'index': idx,
                    'size': info['size'],
                    'quality': info['quality']
                })

        # 각 타입별로 크기 순 정렬 (큰 것부터)
        for ftype in frames_by_type:
            frames_by_type[ftype].sort(key=lambda x: x['size'], reverse=True)

        # 리스트 항목 생성
        # 헤더
        header = QListWidgetItem("=" * 45)
        header.setFlags(Qt.NoItemFlags)  # 클릭 불가
        self.stats_list.addItem(header)

        title = QListWidgetItem("프레임 품질 순위 (용량 기준)")
        title.setFlags(Qt.NoItemFlags)
        title.setFont(QFont("Monospace", 11, QFont.Bold))
        self.stats_list.addItem(title)

        header2 = QListWidgetItem("=" * 45)
        header2.setFlags(Qt.NoItemFlags)
        self.stats_list.addItem(header2)

        spacer = QListWidgetItem("")
        spacer.setFlags(Qt.NoItemFlags)
        self.stats_list.addItem(spacer)

        for ftype, label, color_emoji in [('I', 'I-FRAME', '🟢'),
                                          ('P', 'P-FRAME', '🔵'),
                                          ('B', 'B-FRAME', '🟠')]:
            frames = frames_by_type[ftype]

            # 타입 헤더
            type_header = QListWidgetItem(f"{color_emoji} {label} TOP 10 (최고 품질)")
            type_header.setFlags(Qt.NoItemFlags)
            type_header.setFont(QFont("Monospace", 10, QFont.Bold))
            self.stats_list.addItem(type_header)

            divider = QListWidgetItem("-" * 45)
            divider.setFlags(Qt.NoItemFlags)
            self.stats_list.addItem(divider)

            if not frames:
                no_data = QListWidgetItem("  (없음)")
                no_data.setFlags(Qt.NoItemFlags)
                self.stats_list.addItem(no_data)
            else:
                # 상위 10개만
                top_frames = frames[:10]

                # 평균 대비 계산
                avg_size = self.avg_sizes.get(ftype, 1)

                for rank, frame in enumerate(top_frames, 1):
                    idx = frame['index']
                    size = frame['size']
                    quality = frame['quality']

                    size_kb = size / 1024
                    ratio = (size / avg_size) * 100 if avg_size > 0 else 100
                    time_str = self.format_time_short(idx)

                    if quality is not None:
                        text = f"  {rank:2d}. {time_str} | {size_kb:7.2f}KB ({ratio:5.1f}%) QP:{quality}"
                    else:
                        text = f"  {rank:2d}. {time_str} | {size_kb:7.2f}KB ({ratio:5.1f}%)"

                    item = QListWidgetItem(text)
                    item.setData(Qt.UserRole, idx)  # 프레임 번호 저장
                    self.stats_list.addItem(item)

            # 간격
            spacer = QListWidgetItem("")
            spacer.setFlags(Qt.NoItemFlags)
            self.stats_list.addItem(spacer)

        # QP 기준 정렬 (QP 낮을수록 좋음)
        if any(f.get('quality') is not None for f in self.frame_info):
            qp_header = QListWidgetItem("=" * 45)
            qp_header.setFlags(Qt.NoItemFlags)
            self.stats_list.addItem(qp_header)

            qp_title = QListWidgetItem("QP 기준 품질 순위 (낮을수록 좋음)")
            qp_title.setFlags(Qt.NoItemFlags)
            qp_title.setFont(QFont("Monospace", 11, QFont.Bold))
            self.stats_list.addItem(qp_title)

            qp_header2 = QListWidgetItem("=" * 45)
            qp_header2.setFlags(Qt.NoItemFlags)
            self.stats_list.addItem(qp_header2)

            spacer = QListWidgetItem("")
            spacer.setFlags(Qt.NoItemFlags)
            self.stats_list.addItem(spacer)

            for ftype, label, color_emoji in [('I', 'I-FRAME', '🟢'),
                                              ('P', 'P-FRAME', '🔵'),
                                              ('B', 'B-FRAME', '🟠')]:
                frames = [f for f in frames_by_type[ftype] if f['quality'] is not None]

                if frames:
                    # QP 순 정렬 (낮은 것부터)
                    frames.sort(key=lambda x: x['quality'])

                    type_header = QListWidgetItem(f"{color_emoji} {label} TOP 10 (QP 기준)")
                    type_header.setFlags(Qt.NoItemFlags)
                    type_header.setFont(QFont("Monospace", 10, QFont.Bold))
                    self.stats_list.addItem(type_header)

                    divider = QListWidgetItem("-" * 45)
                    divider.setFlags(Qt.NoItemFlags)
                    self.stats_list.addItem(divider)

                    top_frames = frames[:10]

                    for rank, frame in enumerate(top_frames, 1):
                        idx = frame['index']
                        size = frame['size']
                        quality = frame['quality']

                        size_kb = size / 1024
                        time_str = self.format_time_short(idx)

                        text = f"  {rank:2d}. {time_str} | QP:{quality:2d} ({size_kb:7.2f}KB)"

                        item = QListWidgetItem(text)
                        item.setData(Qt.UserRole, idx)  # 프레임 번호 저장
                        self.stats_list.addItem(item)

                    # 간격
                    spacer = QListWidgetItem("")
                    spacer.setFlags(Qt.NoItemFlags)
                    self.stats_list.addItem(spacer)

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

        # 프레임 품질 분석
        self.statusBar().showMessage('프레임 분석 중...', 0)
        QApplication.processEvents()  # UI 업데이트

        self.frame_info, self.avg_sizes = self.analyze_frame_quality(video_path)

        # 통계 표시
        self.update_stats_display()

        self.statusBar().showMessage('', 0)

        # 10ms 단위로 슬라이더 설정 (1초 = 100단위)
        total_time_ms = int((self.total_frames / self.fps) * 100)

        self.timeline_slider.setMaximum(total_time_ms)
        self.timeline_slider.setEnabled(True)
        self.timeline_slider.setValue(0)
        self.capture_button.setEnabled(True)

        self.show_frame(0)

    def show_frame(self, frame_number):
        if not self.video_capture:
            return

        ret = False
        frame = None

        try:
            # 뒤로 가면 리셋
            if frame_number < self.last_frame_number:
                print(f"[INFO] 뒤로 이동: {self.last_frame_number} -> {frame_number}")
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 10))
                self.last_frame_number = max(-1, frame_number - 11)

            # 5프레임 이내면 순차
            if 0 <= frame_number - self.last_frame_number <= 5:
                for i in range(self.last_frame_number + 1, frame_number + 1):
                    ret, frame = self.video_capture.read()
                    if not ret:
                        print(f"[ERROR] 순차 읽기 실패 at frame {i}")
                        break
            else:
                # 멀리 점프: seek 후 정확한 위치까지 순차 읽기
                seek_target = max(0, frame_number - 10)
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, seek_target)

                for i in range(seek_target, frame_number + 1):
                    ret, frame = self.video_capture.read()
                    if not ret:
                        print(f"[ERROR] seek 후 순차 읽기 실패 at frame {i}")
                        break

            if not ret or frame is None:
                print(f"[ERROR] 프레임 {frame_number} 읽기 완전 실패")
                return

            self.current_frame = frame
            self.last_frame_number = frame_number

            # 프레임 정보 가져오기
            frame_type = '?'
            frame_size = 0
            quality = None
            color = '#757575'

            if 0 <= frame_number < len(self.frame_info):
                info = self.frame_info[frame_number]
                frame_type = info['type']
                frame_size = info['size']
                quality = info['quality']

                # 해당 타입의 평균 대비 퀄리티 비율 계산
                avg_size = self.avg_sizes.get(frame_type, 1)
                quality_ratio = (frame_size / avg_size) * 100 if avg_size > 0 else 100

                # 색상 결정
                if frame_type == 'I':
                    color = '#4CAF50'  # 초록
                elif frame_type == 'P':
                    color = '#2196F3'  # 파랑
                elif frame_type == 'B':
                    # B는 평균 대비 크기로 색상 세분화
                    if quality_ratio > 120:
                        color = '#FFA726'  # 밝은 오렌지 (좋은 B)
                    elif quality_ratio > 80:
                        color = '#FF9800'  # 오렌지 (보통 B)
                    else:
                        color = '#F57C00'  # 어두운 오렌지 (낮은 B)

            print(f"[DEBUG] 프레임 {frame_number} 추출 완료 ({frame_type}, {frame_size}B)")

            # OpenCV BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
            self.video_label.resize(pixmap.size())

            # 타임 업데이트 (타입 정보 + 크기 + QP 포함)
            current_time = frame_number / self.fps if self.fps > 0 else 0
            total_time = self.total_frames / self.fps if self.fps > 0 else 0

            if self.frame_info:  # 프레임 분석이 성공했을 때
                size_kb = frame_size / 1024

                # QP 값이 있으면 표시, 없으면 생략
                if quality is not None:
                    qp_text = f", QP:{quality}"
                else:
                    qp_text = ""

                # 평균 대비 퍼센트
                avg_size = self.avg_sizes.get(frame_type, 1)
                quality_ratio = (frame_size / avg_size) * 100 if avg_size > 0 else 100

                self.time_label.setText(
                    f'{self.format_time(current_time)} / {self.format_time(total_time)} '
                    f'<span style="color: {color}; font-weight: bold;">'
                    f'● {frame_type} ({size_kb:.1f}KB, {quality_ratio:.0f}%{qp_text})</span>'
                )
            else:
                self.time_label.setText(
                    f'{self.format_time(current_time)} / {self.format_time(total_time)}'
                )

        except Exception as e:
            print(f"[CRASH] show_frame 에러: {e}")
            import traceback
            traceback.print_exc()

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}'

    def on_slider_change(self, value):
        # 슬라이더 값(10ms 단위)을 프레임 번호로 변환
        time_seconds = value / 100.0  # 10ms 단위 -> 초
        frame_number = int(time_seconds * self.fps)
        frame_number = min(frame_number, self.total_frames - 1)
        self.show_frame(frame_number)

    def keyPressEvent(self, event):
        if not self.timeline_slider.isEnabled():
            return

        current_value = self.timeline_slider.value()

        if event.key() == Qt.Key_Left:
            # 왼쪽 화살표: 10ms 뒤로 (1단위)
            new_value = max(0, current_value - 1)
            self.timeline_slider.setValue(new_value)
        elif event.key() == Qt.Key_Right:
            # 오른쪽 화살표: 10ms 앞으로 (1단위)
            new_value = min(self.timeline_slider.maximum(), current_value + 1)
            self.timeline_slider.setValue(new_value)
        elif event.key() == Qt.Key_Up:
            # 위쪽 화살표: 1초 앞으로 (100단위)
            new_value = min(self.timeline_slider.maximum(), current_value + 100)
            self.timeline_slider.setValue(new_value)
        elif event.key() == Qt.Key_Down:
            # 아래쪽 화살표: 1초 뒤로 (100단위)
            new_value = max(0, current_value - 100)
            self.timeline_slider.setValue(new_value)
        else:
            super().keyPressEvent(event)

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

                self.statusBar().showMessage(f'프레임 저장 완료: {save_path}', 1500)
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