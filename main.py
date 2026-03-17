import json
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QMessageBox, QScrollArea, QSplitter, QListWidget, QListWidgetItem, QTabWidget)


VERSION = "20260317"

def analyze_sharpness_chunk(args):
    """청크 단위로 선명도 분석 (별도 프로세스)"""
    video_path, chunk_indices = args

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    results = []

    for idx in chunk_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            results.append({
                'frame_index': idx,
                'sharpness': sharpness
            })

    cap.release()
    return results


class VideoFrameExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_capture = None
        self.current_frame = None
        self.total_frames = 0
        self.fps = 0
        self.video_path = None
        self.last_frame_number = -1
        self.frame_info = []
        self.avg_sizes = {}
        self.sharpness_metrics = []

        self.init_ui()
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        self.setWindowTitle('비디오 프레임 추출기')
        self.setGeometry(100, 100, 1800, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Horizontal)

        # 왼쪽: 비디오 영역
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignCenter)

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

        self.time_label = QLabel('00:00:00.000 / 00:00:00.000')
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.time_label)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.timeline_slider)

        control_layout = QHBoxLayout()

        self.open_button = QPushButton('파일 열기')
        self.open_button.clicked.connect(self.open_file)
        control_layout.addWidget(self.open_button)

        self.capture_button = QPushButton('캡처 (PNG 저장)')
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

        # 오른쪽: 통계 영역
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.tab_widget = QTabWidget()

        self.size_list = QListWidget()
        self.setup_list_widget(self.size_list)
        self.tab_widget.addTab(self.size_list, "📦 용량 기준")

        self.sharpness_list = QListWidget()
        self.setup_list_widget(self.sharpness_list)
        self.tab_widget.addTab(self.sharpness_list, "🔍 선명도 기준")

        self.reference_list = QListWidget()
        self.setup_list_widget(self.reference_list)
        self.tab_widget.addTab(self.reference_list, "🎯 참조 프레임")

        right_layout.addWidget(self.tab_widget)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.setAcceptDrops(True)

    def setup_list_widget(self, list_widget):
        list_widget.setMinimumWidth(450)
        font = QFont("SF Mono")
        font.setStyleHint(QFont.TypeWriter)
        font.setPointSize(10)
        list_widget.setFont(font)
        list_widget.setStyleSheet("""
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
        list_widget.itemClicked.connect(self.on_stats_item_clicked)
        list_widget.currentItemChanged.connect(self.on_stats_item_changed)

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
        """비디오의 모든 프레임 타입, 크기, QP, 참조여부 분석"""
        cmd = [
            'ffprobe',
            '-select_streams', 'v:0',
            '-show_frames',
            '-show_entries', 'frame=pict_type,pkt_size,quality,key_frame',
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
                key_frame = int(frame.get('key_frame', 0))

                if quality is not None and not has_quality:
                    has_quality = True

                is_reference = (frame_type == 'I' or key_frame == 1)

                info = {
                    'type': frame_type,
                    'size': frame_size,
                    'quality': quality,
                    'is_reference': is_reference,
                    'key_frame': key_frame
                }
                frame_info.append(info)

            i_count = sum(1 for f in frame_info if f['type'] == 'I')
            p_count = sum(1 for f in frame_info if f['type'] == 'P')
            b_count = sum(1 for f in frame_info if f['type'] == 'B')
            ref_count = sum(1 for f in frame_info if f['is_reference'])

            print(f"[INFO] 프레임 분석 완료: I={i_count}, P={p_count}, B={b_count}, 참조={ref_count}")

            if has_quality:
                print(f"[INFO] QP 값 지원됨")
            else:
                print(f"[INFO] QP 값 미지원")

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

            # 추가 참조 프레임 탐지
            for info in frame_info:
                if info['type'] in ['P', 'B'] and not info['is_reference']:
                    avg = avg_sizes.get(info['type'], 0)
                    if avg > 0 and info['size'] > avg * 1.5:
                        info['is_reference'] = True

            ref_count = sum(1 for f in frame_info if f['is_reference'])
            print(f"[INFO] 크기 분석 후 참조 프레임: {ref_count}개")

            if avg_sizes:
                print(f"[INFO] 평균 크기 - I: {avg_sizes.get('I', 0):.0f}B, "
                      f"P: {avg_sizes.get('P', 0):.0f}B, "
                      f"B: {avg_sizes.get('B', 0):.0f}B")

            # 선명도 병렬 분석
            print("[INFO] 선명도 병렬 분석 시작...")
            sharpness_metrics = self.analyze_sharpness_parallel(video_path, frame_info)

            return frame_info, avg_sizes, sharpness_metrics

        except Exception as e:
            print(f"[ERROR] ffprobe 실패: {e}")
            return [], {}, []

    def analyze_sharpness_parallel(self, video_path, frame_info):
        """멀티프로세싱으로 선명도 분석"""

        # I, P, B 프레임만 필터링
        target_indices = [i for i, info in enumerate(frame_info)
                          if info['type'] in ['I', 'P', 'B']]

        if not target_indices:
            print("[WARN] 분석할 프레임이 없음")
            return []

        print(f"[INFO] {len(target_indices)}개 프레임 병렬 분석 중...")

        # CPU 코어 수
        num_processes = min(cpu_count(), 4)

        # 청크 나누기
        chunk_size = max(1, len(target_indices) // num_processes)
        chunks = []

        for i in range(num_processes):
            start = i * chunk_size
            if i < num_processes - 1:
                end = start + chunk_size
            else:
                end = len(target_indices)

            chunk_indices = target_indices[start:end]

            if chunk_indices:
                # 절대 경로로 변환
                abs_path = str(Path(video_path).resolve())
                chunks.append((abs_path, chunk_indices))

        print(f"[INFO] {len(chunks)}개 청크로 분할")

        # 병렬 처리
        try:
            with Pool(processes=len(chunks)) as pool:
                results = pool.map(analyze_sharpness_chunk, chunks)

            # 결과 병합
            all_metrics = []
            for chunk_result in results:
                all_metrics.extend(chunk_result)

            # 프레임 인덱스 순으로 정렬
            all_metrics.sort(key=lambda x: x['frame_index'])

            print(f"[INFO] 병렬 분석 완료: {len(all_metrics)}개 프레임")

            return all_metrics

        except Exception as e:
            print(f"[ERROR] 병렬 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return []

    def format_time_short(self, frame_number):
        if self.fps == 0:
            return "00:00.000"
        seconds = frame_number / self.fps
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f'{minutes:02d}:{secs:06.3f}'

    def on_stats_item_clicked(self, item):
        frame_number = item.data(Qt.UserRole)
        if frame_number is not None:
            print(f"[INFO] 프레임 {frame_number}로 이동 ({self.format_time_short(frame_number)})")
            self.timeline_slider.setValue(frame_number)

    def on_stats_item_changed(self, current, previous):
        if current is not None:
            frame_number = current.data(Qt.UserRole)
            if frame_number is not None:
                print(f"[INFO] 프레임 {frame_number}로 이동 ({self.format_time_short(frame_number)})")
                self.timeline_slider.setValue(frame_number)

    def update_reference_stats(self):
        self.reference_list.clear()

        if not self.frame_info:
            item = QListWidgetItem("프레임 분석 데이터가 없습니다.")
            self.reference_list.addItem(item)
            return

        ref_frames = []
        for idx, info in enumerate(self.frame_info):
            if info.get('is_reference', False):
                ref_frames.append({
                    'index': idx,
                    'type': info['type'],
                    'size': info['size'],
                    'quality': info['quality']
                })

        header = QListWidgetItem("=" * 65)
        header.setFlags(Qt.NoItemFlags)
        self.reference_list.addItem(header)

        title = QListWidgetItem(f"참조 프레임 목록 (총 {len(ref_frames)}개)")
        title.setFlags(Qt.NoItemFlags)
        title.setFont(QFont("SF Mono", 12, QFont.Bold))
        self.reference_list.addItem(title)

        subtitle = QListWidgetItem("(다른 프레임의 기점이 되는 프레임)")
        subtitle.setFlags(Qt.NoItemFlags)
        self.reference_list.addItem(subtitle)

        header2 = QListWidgetItem("=" * 65)
        header2.setFlags(Qt.NoItemFlags)
        self.reference_list.addItem(header2)

        spacer = QListWidgetItem("")
        spacer.setFlags(Qt.NoItemFlags)
        self.reference_list.addItem(spacer)

        for rank, frame in enumerate(ref_frames, 1):
            idx = frame['index']
            ftype = frame['type']
            size = frame['size']
            quality = frame['quality']

            size_kb = size / 1024
            time_str = self.format_time_short(idx)

            avg_size = self.avg_sizes.get(ftype, 1)
            ratio = (size / avg_size) * 100 if avg_size > 0 else 100

            if ftype == 'I':
                emoji = '⭐🟢'
            elif ftype == 'P':
                emoji = '⭐🔵'
            elif ftype == 'B':
                emoji = '⭐🟠'
            else:
                emoji = '⭐⚪'

            if quality is not None:
                text = f"  {rank:4d}. {time_str} | {emoji}{ftype} {size_kb:8.4f}KB ({ratio:6.2f}%) QP:{quality}"
            else:
                text = f"  {rank:4d}. {time_str} | {emoji}{ftype} {size_kb:8.4f}KB ({ratio:6.2f}%)"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            self.reference_list.addItem(item)

    def update_size_stats(self):
        self.size_list.clear()

        if not self.frame_info:
            item = QListWidgetItem("프레임 분석 데이터가 없습니다.")
            self.size_list.addItem(item)
            return

        header = QListWidgetItem("=" * 65)
        header.setFlags(Qt.NoItemFlags)
        self.size_list.addItem(header)

        all_frames_title = QListWidgetItem("전체 프레임 TOP 15 (용량 기준)")
        all_frames_title.setFlags(Qt.NoItemFlags)
        all_frames_title.setFont(QFont("SF Mono", 11, QFont.Bold))
        self.size_list.addItem(all_frames_title)

        header2 = QListWidgetItem("=" * 65)
        header2.setFlags(Qt.NoItemFlags)
        self.size_list.addItem(header2)

        all_frames_sorted = []
        for idx, info in enumerate(self.frame_info):
            all_frames_sorted.append({
                'index': idx,
                'type': info['type'],
                'size': info['size'],
                'quality': info['quality'],
                'is_reference': info.get('is_reference', False)
            })
        all_frames_sorted.sort(key=lambda x: x['size'], reverse=True)

        for rank, frame in enumerate(all_frames_sorted[:15], 1):
            idx = frame['index']
            ftype = frame['type']
            size = frame['size']
            quality = frame['quality']
            is_ref = frame['is_reference']

            size_kb = size / 1024
            time_str = self.format_time_short(idx)

            if is_ref:
                emoji = {'I': '⭐🟢', 'P': '⭐🔵', 'B': '⭐🟠'}.get(ftype, '⭐⚪')
            else:
                emoji = {'I': '🟢', 'P': '🔵', 'B': '🟠'}.get(ftype, '⚪')

            if quality is not None:
                text = f"  {rank:2d}. {time_str} | {emoji}{ftype} {size_kb:10.4f}KB QP:{quality}"
            else:
                text = f"  {rank:2d}. {time_str} | {emoji}{ftype} {size_kb:10.4f}KB"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            self.size_list.addItem(item)

        self._add_type_based_stats(self.size_list)

    def update_sharpness_stats(self):
        self.sharpness_list.clear()

        if not self.sharpness_metrics:
            item = QListWidgetItem("선명도 분석 데이터가 없습니다.")
            self.sharpness_list.addItem(item)
            return

        sorted_metrics = sorted(self.sharpness_metrics, key=lambda x: x['sharpness'], reverse=True)

        header = QListWidgetItem("=" * 65)
        header.setFlags(Qt.NoItemFlags)
        self.sharpness_list.addItem(header)

        title = QListWidgetItem("전체 프레임 선명도 순위")
        title.setFlags(Qt.NoItemFlags)
        title.setFont(QFont("SF Mono", 12, QFont.Bold))
        self.sharpness_list.addItem(title)

        subtitle = QListWidgetItem("(높을수록 선명함)")
        subtitle.setFlags(Qt.NoItemFlags)
        self.sharpness_list.addItem(subtitle)

        header2 = QListWidgetItem("=" * 65)
        header2.setFlags(Qt.NoItemFlags)
        self.sharpness_list.addItem(header2)

        spacer = QListWidgetItem("")
        spacer.setFlags(Qt.NoItemFlags)
        self.sharpness_list.addItem(spacer)

        for rank, metrics in enumerate(sorted_metrics, 1):
            idx = metrics['frame_index']
            sharpness = metrics['sharpness']
            time_str = self.format_time_short(idx)

            ftype = self.frame_info[idx]['type']
            size = self.frame_info[idx]['size']
            size_kb = size / 1024
            is_ref = self.frame_info[idx].get('is_reference', False)

            avg_size = self.avg_sizes.get(ftype, 1)
            ratio = (size / avg_size) * 100 if avg_size > 0 else 100

            if is_ref:
                emoji = {'I': '⭐🟢', 'P': '⭐🔵', 'B': '⭐🟠'}.get(ftype, '⭐⚪')
            else:
                emoji = {'I': '🟢', 'P': '🔵', 'B': '🟠'}.get(ftype, '⚪')

            text = f"  {rank:4d}. {time_str} | {emoji}{ftype} 선명:{sharpness:8.4f} {size_kb:8.4f}KB ({ratio:6.2f}%)"

            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            self.sharpness_list.addItem(item)

    def _add_type_based_stats(self, list_widget):
        frames_by_type = {'I': [], 'P': [], 'B': []}

        for idx, info in enumerate(self.frame_info):
            frame_type = info['type']
            if frame_type in frames_by_type:
                frames_by_type[frame_type].append({
                    'index': idx,
                    'size': info['size'],
                    'quality': info['quality'],
                    'is_reference': info.get('is_reference', False)
                })

        frames_by_type['I'].sort(key=lambda x: x['size'], reverse=True)
        frames_by_type['P'].sort(key=lambda x: x['size'], reverse=True)
        frames_by_type['B'].sort(key=lambda x: x['size'])

        spacer = QListWidgetItem("")
        spacer.setFlags(Qt.NoItemFlags)
        list_widget.addItem(spacer)
        spacer = QListWidgetItem("")
        spacer.setFlags(Qt.NoItemFlags)
        list_widget.addItem(spacer)

        header = QListWidgetItem("=" * 65)
        header.setFlags(Qt.NoItemFlags)
        list_widget.addItem(header)

        title = QListWidgetItem("타입별 프레임 순위")
        title.setFlags(Qt.NoItemFlags)
        title.setFont(QFont("SF Mono", 11, QFont.Bold))
        list_widget.addItem(title)

        header2 = QListWidgetItem("=" * 65)
        header2.setFlags(Qt.NoItemFlags)
        list_widget.addItem(header2)

        spacer = QListWidgetItem("")
        spacer.setFlags(Qt.NoItemFlags)
        list_widget.addItem(spacer)

        for ftype, label, color_emoji, desc in [
            ('I', 'I-FRAME', '🟢', '용량 큰 순'),
            ('P', 'P-FRAME', '🔵', '용량 큰 순'),
            ('B', 'B-FRAME', '🟠', '용량 작은 순 (원본에 가까움)')
        ]:
            frames = frames_by_type[ftype]

            type_header = QListWidgetItem(f"{color_emoji} {label} TOP 50 ({desc})")
            type_header.setFlags(Qt.NoItemFlags)
            type_header.setFont(QFont("SF Mono", 10, QFont.Bold))
            list_widget.addItem(type_header)

            divider = QListWidgetItem("-" * 65)
            divider.setFlags(Qt.NoItemFlags)
            list_widget.addItem(divider)

            if not frames:
                no_data = QListWidgetItem("  (없음)")
                no_data.setFlags(Qt.NoItemFlags)
                list_widget.addItem(no_data)
            else:
                top_frames = frames[:50]
                avg_size = self.avg_sizes.get(ftype, 1)

                for rank, frame in enumerate(top_frames, 1):
                    idx = frame['index']
                    size = frame['size']
                    quality = frame['quality']
                    is_ref = frame['is_reference']

                    size_kb = size / 1024
                    ratio = (size / avg_size) * 100 if avg_size > 0 else 100
                    time_str = self.format_time_short(idx)

                    ref_mark = '⭐' if is_ref else '  '

                    if quality is not None:
                        text = f"{ref_mark}{rank:2d}. {time_str} | {size_kb:10.4f}KB ({ratio:6.2f}%) QP:{quality}"
                    else:
                        text = f"{ref_mark}{rank:2d}. {time_str} | {size_kb:10.4f}KB ({ratio:6.2f}%)"

                    item = QListWidgetItem(text)
                    item.setData(Qt.UserRole, idx)
                    list_widget.addItem(item)

            spacer = QListWidgetItem("")
            spacer.setFlags(Qt.NoItemFlags)
            list_widget.addItem(spacer)

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

        self.statusBar().showMessage('프레임 분석 중...', 0)
        QApplication.processEvents()

        self.frame_info, self.avg_sizes, self.sharpness_metrics = self.analyze_frame_quality(video_path)

        self.update_size_stats()
        self.update_sharpness_stats()
        self.update_reference_stats()

        self.statusBar().showMessage('', 0)

        self.timeline_slider.setMaximum(self.total_frames - 1)
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
            if frame_number < self.last_frame_number:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 10))
                self.last_frame_number = max(-1, frame_number - 11)

            if 0 <= frame_number - self.last_frame_number <= 5:
                for i in range(self.last_frame_number + 1, frame_number + 1):
                    ret, frame = self.video_capture.read()
                    if not ret:
                        break
            else:
                seek_target = max(0, frame_number - 10)
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, seek_target)

                for i in range(seek_target, frame_number + 1):
                    ret, frame = self.video_capture.read()
                    if not ret:
                        break

            if not ret or frame is None:
                return

            self.current_frame = frame
            self.last_frame_number = frame_number

            frame_type = '?'
            frame_size = 0
            quality = None
            is_reference = False
            color = '#757575'

            if 0 <= frame_number < len(self.frame_info):
                info = self.frame_info[frame_number]
                frame_type = info['type']
                frame_size = info['size']
                quality = info['quality']
                is_reference = info.get('is_reference', False)

                avg_size = self.avg_sizes.get(frame_type, 1)
                quality_ratio = (frame_size / avg_size) * 100 if avg_size > 0 else 100

                if frame_type == 'I':
                    color = '#4CAF50'
                elif frame_type == 'P':
                    color = '#2196F3'
                elif frame_type == 'B':
                    if quality_ratio > 120:
                        color = '#FFA726'
                    elif quality_ratio > 80:
                        color = '#FF9800'
                    else:
                        color = '#F57C00'

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
            self.video_label.resize(pixmap.size())

            current_time = frame_number / self.fps if self.fps > 0 else 0
            total_time = self.total_frames / self.fps if self.fps > 0 else 0

            if self.frame_info:
                size_kb = frame_size / 1024

                if quality is not None:
                    qp_text = f", QP:{quality}"
                else:
                    qp_text = ""

                avg_size = self.avg_sizes.get(frame_type, 1)
                quality_ratio = (frame_size / avg_size) * 100 if avg_size > 0 else 100

                ref_text = ' [참조⭐]' if is_reference else ''

                self.time_label.setText(
                    f'{self.format_time(current_time)} / {self.format_time(total_time)} '
                    f'<span style="color: {color}; font-weight: bold;">'
                    f'● {frame_type} ({size_kb:.4f}KB, {quality_ratio:.2f}%{qp_text}){ref_text}</span>'
                )
            else:
                self.time_label.setText(
                    f'{self.format_time(current_time)} / {self.format_time(total_time)}'
                )

        except Exception as e:
            print(f"[ERROR] show_frame: {e}")
            import traceback
            traceback.print_exc()

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f'{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}'

    def on_slider_change(self, value):
        frame_number = min(value, self.total_frames - 1)
        self.show_frame(frame_number)

    def keyPressEvent(self, event):
        if not self.timeline_slider.isEnabled():
            return

        current_value = self.timeline_slider.value()

        if event.key() == Qt.Key_Left:
            new_value = max(0, current_value - 1)
            self.timeline_slider.setValue(new_value)
        elif event.key() == Qt.Key_Right:
            new_value = min(self.timeline_slider.maximum(), current_value + 1)
            self.timeline_slider.setValue(new_value)
        elif event.key() == Qt.Key_Up:
            jump = int(self.fps) if self.fps > 0 else 30
            new_value = min(self.timeline_slider.maximum(), current_value + jump)
            self.timeline_slider.setValue(new_value)
        elif event.key() == Qt.Key_Down:
            jump = int(self.fps) if self.fps > 0 else 30
            new_value = max(0, current_value - jump)
            self.timeline_slider.setValue(new_value)
        else:
            super().keyPressEvent(event)

    def capture_frame(self):
        if self.current_frame is None:
            QMessageBox.warning(self, '오류', '캡처할 프레임이 없습니다.')
            return

        if self.video_path:
            video_filename = Path(self.video_path).stem
            default_name = f'{video_filename}.webp'
        else:
            current_frame_num = self.timeline_slider.value()
            default_name = f'frame_{current_frame_num:06d}.webp'

        project_dir = Path(__file__).parent / VERSION
        project_dir.mkdir(exist_ok=True)
        save_path, _ = QFileDialog.getSaveFileName(
            self, '프레임 저장', f'{project_dir}/{default_name}', 'webp Files (*.webp)'
        )

        if save_path:
            try:
                # 1. OpenCV BGR을 RGB로 변환 및 PIL 객체 생성
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # 2. 리사이징 로직 추가 (최대 384px)
                max_size = 384
                # thumbnail은 원본 비율을 유지하며, 이미지의 가로/세로 중 큰 쪽을 max_size에 맞춥니다.
                # 이미 384보다 작다면 아무 작업도 하지 않습니다.
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # 3. 저장 (WebP)
                pil_image.save(save_path, 'WebP', quality=75, method=6)

                self.statusBar().showMessage(f'프레임 저장 완료 (리사이징 적용): {save_path}', 1500)
            except Exception as e:
                QMessageBox.critical(self, '오류', f'저장 실패:\n{str(e)}')

        # self.capture_frame_png()

    def capture_frame_png(self):
        if self.current_frame is None:
            QMessageBox.warning(self, '오류', '캡처할 프레임이 없습니다.')
            return

        if self.video_path:
            video_filename = Path(self.video_path).stem
            default_name = f'{video_filename}.png'
        else:
            current_frame_num = self.timeline_slider.value()
            default_name = f'frame_{current_frame_num:06d}.png'

        project_dir = Path(__file__).parent / VERSION
        project_dir.mkdir(exist_ok=True)
        save_path, _ = QFileDialog.getSaveFileName(
            self, '프레임 저장', f'{project_dir}/{default_name}', 'png Files (*.png)'
        )

        if save_path:
            try:
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image.save(save_path, 'PNG')

                self.statusBar().showMessage(f'PNG 프레임 저장 완료: {save_path}', 1500)
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
    import multiprocessing

    multiprocessing.freeze_support()
    main()