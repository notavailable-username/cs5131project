import sys
import os
import csv
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
    QPushButton, QComboBox, QWidget, QFileDialog, QLineEdit, 
    QTableWidget, QTableWidgetItem, QMessageBox, QStackedLayout,
    QSizePolicy, QDockWidget, QListWidget, QToolButton, QStyle
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon
from PyQt6.QtCore import Qt, QEvent

class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Labeling Tool")
        self.setMinimumSize(1200, 800)

        # Initialize state variables
        self.image_dir = None
        self.image_list = []
        self.current_image_index = -1
        self.current_image = None
        self.current_pixmap = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.bounding_boxes = []
        self.temp_bounding_box = []
        self.classes = []  # List of tuples: (class_id, class_name)
        self.original_classes = []
        self.edit_mode = "NIL"
        self.selected_rows = set()

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left Panel: Collapsible Toolbar with Hamburger Menu
        self.dock_widget = QDockWidget(self)
        self.dock_widget.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.dock_widget.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)

        # Create a widget to hold the hamburger button and image list
        dock_content = QWidget()
        dock_layout = QVBoxLayout()
        dock_content.setLayout(dock_layout)

        # Hamburger Menu Button
        self.hamburger_button = QToolButton()
        self.hamburger_button.setText("☰")
        self.hamburger_button.setStyleSheet("""
            QToolButton {
                font-size: 12px;
                padding: 7px;
                color: white;
                background-color: #444;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QToolButton:hover {
                background-color: #666;
            }
            QToolButton:checked {
                background-color: #888;
                border: 1px solid #aaa;
                color: white;
            }
        """)
        self.hamburger_button.setCheckable(True)
        self.hamburger_button.setChecked(False)
        self.hamburger_button.clicked.connect(self.toggle_image_list)
        dock_layout.addWidget(self.hamburger_button, alignment=Qt.AlignmentFlag.AlignTop)

        # Image List Widget
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.load_image_from_list)
        self.image_list_widget.setSizeAdjustPolicy(QListWidget.SizeAdjustPolicy.AdjustToContents)
        self.image_list_widget.setStyleSheet("""
            QListWidget {
                background-color: #333;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:selected {
                background-color: #555;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #444;
            }
        """)
        dock_layout.addWidget(self.image_list_widget)
        self.image_list_widget.setVisible(False)

        self.dock_widget.setWidget(dock_content)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_widget)

        # Center Panel: Image Display and Navigation Buttons
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        center_panel.setLayout(center_layout)

        # Image Display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.start_drawing
        self.image_label.mouseMoveEvent = self.update_drawing
        self.image_label.mouseReleaseEvent = self.finish_drawing
        center_layout.addWidget(self.image_label, stretch=1)

        # Navigation Buttons
        nav_layout = QHBoxLayout()
        prev_button = QPushButton("Previous Image")
        prev_button.clicked.connect(self.load_previous_image)
        nav_layout.addWidget(prev_button)

        next_button = QPushButton("Next Image")
        next_button.clicked.connect(self.load_next_image)
        nav_layout.addWidget(next_button)
        center_layout.addLayout(nav_layout)

        main_layout.addWidget(center_panel)

        # Right Panel: Stacked Layout with Button Container
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)

        # Create button container and stacked layout
        self.stack_layout = QStackedLayout()
        button_container = QHBoxLayout()
        button_container.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Navigation buttons
        self.annotate_button = QPushButton("Annotation")
        self.annotate_button.setCheckable(True)
        self.annotate_button.setAutoExclusive(True)
        self.annotate_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self.annotate_button.setStyleSheet(self.get_button_style())
        
        self.classes_button = QPushButton("Class Management")
        self.classes_button.setCheckable(True)
        self.classes_button.setAutoExclusive(True)
        self.classes_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self.classes_button.setStyleSheet(self.get_button_style())

        button_container.addWidget(self.annotate_button)
        button_container.addWidget(self.classes_button)
        
        # Connect buttons
        self.annotate_button.clicked.connect(lambda: self.stack_layout.setCurrentIndex(0))
        self.classes_button.clicked.connect(lambda: self.stack_layout.setCurrentIndex(1))

        right_layout.addLayout(button_container)
        right_layout.addLayout(self.stack_layout)

        # Create tabs
        self.create_annotation_tab()
        self.create_classes_tab()

        # Initial view
        self.stack_layout.setCurrentIndex(0)
        self.annotate_button.setChecked(True)

        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        pixmap = icon.pixmap(32, 32)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor("#4CAF50"))  # Green color
        painter.end()
        colored_icon = QIcon(pixmap)

        load_dir_action = file_menu.addAction(colored_icon, "Load Image Directory")
        load_dir_action.setShortcut("Ctrl+O")

        load_dir_action.icon = self.style().standardIcon(self.style().StandardPixmap.SP_DirOpenIcon)
        load_dir_action.triggered.connect(self.load_image_directory)

    def eventFilter(self, source, event):
        if (event.type() == QEvent.Type.KeyPress and 
            source is self.class_table and 
            event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            
            selected_rows = sorted({index.row() for index in self.class_table.selectedIndexes()})
            
            if event.key() == Qt.Key.Key_Up:
                self.move_classes(selected_rows, "up")
                return True
                
            elif event.key() == Qt.Key.Key_Down:
                self.move_classes(selected_rows, "down")
                return True
                
        return super().eventFilter(source, event)
    
    def move_classes(self, selected_rows, direction):
        if not selected_rows:
            return

        # Create copy of classes list
        self.original_classes = self.classes.copy()
        new_classes = self.classes.copy()
        
        if direction == "up":
            # Can't move above first row
            if 0 in selected_rows:
                return
                
            # Move each selected row up by swapping with previous
            for row in sorted(selected_rows):
                new_classes[row], new_classes[row-1] = new_classes[row-1], new_classes[row]
                
        elif direction == "down":
            # Can't move below last row
            if (len(new_classes)-1) in selected_rows:
                return
                
            # Move each selected row down by swapping with next
            for row in sorted(selected_rows, reverse=True):
                new_classes[row], new_classes[row+1] = new_classes[row+1], new_classes[row]

        # Update class list with new order
        self.classes = [(idx, name) for idx, (_, name) in enumerate(new_classes)]
        
        # Update annotations and UI
        self.update_annotations(self.original_classes, self.classes)
        self.save_classes_to_csv()
        self.class_search_input.clear()
        
        # Preserve selection after move
        new_selection = [row-1 if direction == "up" else row+1 for row in selected_rows]
        self.class_table.clearSelection()
        for row in new_selection:
            self.class_table.selectRow(row)

    def get_button_style(self):
        return """
            QPushButton {
                padding: 7px;
                color: white;
                background-color: #333;
                border: 1px solid #555;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:checked {
                background-color: #555;
                border: 1px solid #aaa;
                color: white;
            }
            QPushButton:pressed {
                background-color: black;
                border: 1px solid #aaa;
                color: white;
            }
        """

    def toggle_image_list(self):
        is_visible = self.image_list_widget.isVisible()
        self.image_list_widget.setVisible(not is_visible)

    def update_image_list_widget(self):
        self.image_list_widget.clear()
        for image_name in self.image_list:
            self.image_list_widget.addItem(image_name)
        if self.current_image_index >= 0:
            self.image_list_widget.setCurrentRow(self.current_image_index)

    def load_image_from_list(self, item):
        self.current_image_index = self.image_list_widget.row(item)
        self.load_image()

    def create_annotation_tab(self):
        annotate_tab = QWidget()
        annotate_layout = QVBoxLayout()
        annotate_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        annotate_tab.setLayout(annotate_layout)

        # Class Selection Container moved above the table
        class_select_container = QWidget()
        class_select_layout = QVBoxLayout()
        class_select_container.setLayout(class_select_layout)
        class_select_layout.setContentsMargins(0, 0, 0, 0)
        class_select_layout.setSpacing(0)

        # Search Input
        self.class_search_input = QLineEdit()
        self.class_search_input.setPlaceholderText("Search or select class...")
        self.class_search_input.textChanged.connect(self.filter_classes)
        self.class_search_input.setStyleSheet("""
            QLineEdit {
                color: black;
                padding: 7px;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding-right: 25px;
                background-color: white;
            }
        """)
        class_select_layout.addWidget(self.class_search_input)

        # Class Dropdown
        self.class_dropdown = QComboBox()
        self.class_dropdown.setMaximumHeight(150)
        self.class_dropdown.hide()
        self.class_dropdown.setStyleSheet("""
            QComboBox {
                color: black;
                padding: 5px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 4px 4px;
                background-color: white;
            }
            QComboBox QAbstractItemView::item {
                color: black;
                padding: 5px;
                background-color: white;
            }
            QComboBox QAbstractItemView::item:hover {
                color: black;
                padding: 5px;
                background-color: #e0e0e0;
            }
        """)
        self.class_dropdown.activated.connect(self.handle_class_selection)
        class_select_layout.addWidget(self.class_dropdown)

        annotate_layout.addWidget(class_select_container)

        # Annotation Table
        self.annotation_table = QTableWidget()
        self.annotation_table.setColumnCount(5)
        self.annotation_table.setHorizontalHeaderLabels(["Class ID", "X Center", "Y Center", "Width", "Height"])
        self.annotation_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.annotation_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.annotation_table.itemSelectionChanged.connect(self.select_annotation_from_table)
        annotate_layout.addWidget(self.annotation_table)

        # Button Container
        button_container = QHBoxLayout()
        
        # Delete Button
        delete_annotation_button = QPushButton("Delete Selected Annotations")
        delete_annotation_button.setStyleSheet(self.get_button_style())
        delete_annotation_button.clicked.connect(self.delete_selected_annotation)
        button_container.addWidget(delete_annotation_button)

        # Save Button
        save_button = QPushButton("Save Annotations")
        save_button.setStyleSheet(self.get_button_style())
        save_button.clicked.connect(self.save_annotations)
        button_container.addWidget(save_button)

        annotate_layout.addLayout(button_container)

        self.stack_layout.addWidget(annotate_tab)
    
    def filter_classes(self, text):
        self.class_search_input.setProperty("selected_class_id", None)
        self.class_dropdown.clear()
        current_text = text.lower()
        
        for class_id, class_name in self.classes:
            if current_text in class_name.lower() or current_text in str(class_id):
                self.class_dropdown.addItem(f"{class_id}: {class_name}", userData=class_id)
        
        self.class_dropdown.setVisible(len(current_text) > 0 and self.class_dropdown.count() > 0)
    
    def handle_class_selection(self, index):
        if index >= 0:
            class_name = self.class_dropdown.itemText(index)
            class_id = self.class_dropdown.itemData(index)
            self.class_search_input.setText(class_name)
            self.class_search_input.setProperty("selected_class_id", class_id)
            self.class_dropdown.hide()

    def select_annotation_from_table(self):
        selected_rows = set()
        for item in self.annotation_table.selectedItems():
            selected_rows.add(item.row())
        self.selected_rows = selected_rows
        self.update_image_display()

    def delete_selected_annotation(self):
        selected_rows = set()
        for item in self.annotation_table.selectedItems():
            selected_rows.add(item.row())
        
        # Delete in reverse order to maintain correct indices
        for row in sorted(self.selected_rows, reverse=True):
            self.bounding_boxes.pop(row)
            
        self.update_annotation_table()
        self.update_image_display()
        self.refresh_class_table()

    def update_annotation_table(self):
        self.annotation_table.setRowCount(0)
        self.bounding_boxes.sort()
        for idx, bbox in enumerate(self.bounding_boxes):
            class_id, x_center, y_center, width, height = bbox
            self.annotation_table.insertRow(idx)
            self.annotation_table.setItem(idx, 0, QTableWidgetItem(str(int(class_id))))
            self.annotation_table.setItem(idx, 1, QTableWidgetItem(f"{x_center:.6f}"))
            self.annotation_table.setItem(idx, 2, QTableWidgetItem(f"{y_center:.6f}"))
            self.annotation_table.setItem(idx, 3, QTableWidgetItem(f"{width:.6f}"))
            self.annotation_table.setItem(idx, 4, QTableWidgetItem(f"{height:.6f}"))

    def create_classes_tab(self):
        classes_tab = QWidget()
        classes_layout = QVBoxLayout()
        classes_tab.setLayout(classes_layout)

        classes_tab = QWidget()
        classes_layout = QVBoxLayout()
        classes_tab.setLayout(classes_layout)

        # Class Table
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(3)
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setHorizontalHeaderLabels(["Class ID", "Class Name", "Annotations"])
        self.class_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.class_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.class_table.installEventFilter(self)
        classes_layout.addWidget(self.class_table)

        # Add Class Section
        add_class_container = QWidget()
        add_class_layout = QHBoxLayout()
        add_class_container.setLayout(add_class_layout)
        add_class_layout.setContentsMargins(0, 0, 0, 0)
        
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("Enter new class name")
        self.new_class_input.setStyleSheet("""
            QLineEdit {
                color: black;
                padding: 7px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
        """)
        add_class_layout.addWidget(self.new_class_input)

        add_class_button = QPushButton("Add Class")
        add_class_button.setStyleSheet(self.get_button_style())
        add_class_button.clicked.connect(self.add_class)
        add_class_layout.addWidget(add_class_button)
        classes_layout.addWidget(add_class_container)

        # Edit Controls
        edit_controls = QHBoxLayout()
        self.rename_button = QPushButton("Rename Classes")
        self.rename_button.setStyleSheet(self.get_button_style())
        self.rename_button.clicked.connect(lambda: self.toggle_edit_mode("RENAME"))
        edit_controls.addWidget(self.rename_button)

        self.reorder_button = QPushButton("Reorder Classes")
        self.reorder_button.setStyleSheet(self.get_button_style())
        self.reorder_button.clicked.connect(lambda: self.toggle_edit_mode("REORDER"))
        edit_controls.addWidget(self.reorder_button)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.setStyleSheet(self.get_button_style())
        self.confirm_button.clicked.connect(self.confirm_edit)
        self.confirm_button.hide()
        edit_controls.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(self.get_button_style())
        self.cancel_button.clicked.connect(self.cancel_edit)
        self.cancel_button.hide()
        edit_controls.addWidget(self.cancel_button)
        classes_layout.addLayout(edit_controls)

        # Delete Button
        delete_class_button = QPushButton("Delete Selected Class")
        delete_class_button.setStyleSheet(self.get_button_style())
        delete_class_button.clicked.connect(self.delete_class)
        classes_layout.addWidget(delete_class_button)

        self.stack_layout.addWidget(classes_tab)

    def toggle_edit_mode(self, edit_mode):
        if self.image_dir:
            self.edit_mode = edit_mode
            self.class_table.setStyleSheet("")
            match self.edit_mode:
                case "NIL":
                    self.original_classes = self.classes.copy()
                    self.class_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
                    self.rename_button.show()
                    self.reorder_button.show()
                    self.confirm_button.hide()
                    self.cancel_button.hide()
                    
                case "RENAME":
                    self.original_classes = self.classes.copy()
                    self.class_table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
                    for row in range(self.class_table.rowCount()):
                        self.class_table.item(row, 0).setFlags(self.class_table.item(row, 0).flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.class_table.item(row, 1).setFlags(self.class_table.item(row, 1).flags() | Qt.ItemFlag.ItemIsEditable)
                        self.class_table.item(row, 2).setFlags(self.class_table.item(row, 2).flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.rename_button.hide()
                    self.reorder_button.hide()
                    self.confirm_button.show()
                    self.cancel_button.show()

                case "REORDER":
                    self.class_table.setStyleSheet("""
                        QTableWidget {
                            border: 2px dashed #4CAF50;
                        }
                        QTableWidget::item {
                            padding: 5px;
                        }
                    """)
                    self.class_table.setDragDropMode(QTableWidget.DragDropMode.InternalMove)
                    self.original_classes = self.classes.copy()
                    self.class_table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
                    for row in range(self.class_table.rowCount()):
                        self.class_table.item(row, 0).setFlags(self.class_table.item(row, 0).flags() | Qt.ItemFlag.ItemIsEditable)
                        self.class_table.item(row, 1).setFlags(self.class_table.item(row, 1).flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.class_table.item(row, 2).setFlags(self.class_table.item(row, 2).flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.rename_button.hide()
                    self.reorder_button.hide()
                    self.confirm_button.show()
                    self.cancel_button.show()

    def confirm_edit(self):
        try:
            new_classes = []
            for row in range(self.class_table.rowCount()):
                index_item = self.class_table.item(row, 0)
                name_item = self.class_table.item(row, 1)
                if not index_item or not name_item:
                    raise ValueError("Missing values in table")
                
                class_id = int(index_item.text())
                class_name = name_item.text().strip()
                if not class_name:
                    raise ValueError("Class name cannot be empty")
                
                new_classes.append((class_id, class_name))

            # Check for duplicates and continuity
            indexes = sorted([c[0] for c in new_classes])
            if indexes != list(range(len(indexes))):
                raise ValueError("Class indexes must be continuous starting from 0")
            if len(set(indexes)) != len(indexes):
                raise ValueError("Duplicate class indexes found")
            
            if len(set([c[1] for c in new_classes])) != len(new_classes):
                raise ValueError("Duplicate class names found")

            self.classes = sorted(new_classes, key=lambda x: x[0])
            self.save_classes_to_csv()
            if self.edit_mode=="REORDER":
                self.update_annotations(self.original_classes, self.classes)
                self.refresh_class_table()
                self.class_search_input.clear()
            self.update_image_display()
            self.toggle_edit_mode("NIL")
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            self.cancel_edit()

    def cancel_edit(self):
        self.classes = self.original_classes.copy()
        self.refresh_class_table()
        self.toggle_edit_mode("NIL")

    def load_classes_from_csv(self):
        if self.image_dir:
            csv_path = os.path.join(self.image_dir, "classes.csv")
            if os.path.exists(csv_path):
                self.classes = []
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            self.classes.append((int(row[0]), row[1]))
                self.refresh_class_table()
                self.update_class_dropdown()

    def save_classes_to_csv(self):
        if self.image_dir:
            csv_path = os.path.join(self.image_dir, "classes.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for class_id, class_name in self.classes:
                    writer.writerow([class_id, class_name])
            self.refresh_class_table()
            self.update_class_dropdown()

    def add_class(self):
        if self.image_dir:
            class_name = self.new_class_input.text().strip()
            if class_name and not any(c[1] == class_name for c in self.classes):
                class_id = len(self.classes)
                self.classes.append((class_id, class_name))
                self.new_class_input.clear()
                self.save_classes_to_csv()
                self.update_class_dropdown()
                self.refresh_class_table()

    def delete_class(self):
        selected_items = self.class_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        class_id = int(self.class_table.item(row, 0).text())
        class_name = self.class_table.item(row, 1).text()
        self.original_classes = self.classes.copy()

        # Check if class is used in any annotations
        class_used = False
        if self.image_dir:
            for image_name in self.image_list:
                label_path = os.path.join(self.image_dir, os.path.splitext(image_name)[0] + ".txt")
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            if line.startswith(f"{class_id} "):
                                class_used = True
                                break
                    if class_used:
                        break

        if class_used:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Class in Use")
            msg.setText(f"Class '{class_name}' is used in annotations!\n"
                        "Consider renaming instead of deleting.\n"
                        "Delete anyway?")
            msg.setStandardButtons(QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes)
            msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
            answer = msg.exec()

            if answer == QMessageBox.StandardButton.Cancel:
                return

            # Remove all annotations with this class
            for image_name in self.image_list:
                label_path = os.path.join(self.image_dir, os.path.splitext(image_name)[0] + ".txt")
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    filtered_lines = []
                    for line in lines:
                        if not line.startswith(f"{class_id} "):
                            filtered_lines.append(line)
                    
                    with open(label_path, 'w') as f:
                        f.writelines(filtered_lines)

        # Remove class from list and update UI
        new_classes = [(idx, name) for idx, (_, name) in enumerate([c for c in self.classes if c[0] != class_id])]
        self.classes = new_classes

        self.save_classes_to_csv()
        self.update_class_dropdown()

        self.update_annotations(self.original_classes, self.classes, deleted_class_id=class_id)
        self.refresh_class_table()
        
        self.class_search_input.clear()

        # Reload current image if needed
        if self.current_image_index >= 0:
            self.load_image()

    def count_annotations_per_class(self):
        class_counts = {class_id: 0 for class_id, _ in self.classes}
        if not self.image_dir:
            return class_counts
        
        for image_name in self.image_list:
            label_path = os.path.join(self.image_dir, os.path.splitext(image_name)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id in class_counts:
                                class_counts[class_id] += 1
        return class_counts

    def refresh_class_table(self):
        self.class_table.setRowCount(0)
        class_counts = self.count_annotations_per_class()
        total_annotations = sum(class_counts.values())
        num_classes = len(self.classes)
        average_percent = 100 / num_classes if num_classes > 0 else 0

        for class_id, class_name in sorted(self.classes, key=lambda x: x[0]):
            row_position = self.class_table.rowCount()
            self.class_table.insertRow(row_position)
            self.class_table.setItem(row_position, 0, QTableWidgetItem(str(class_id)))
            self.class_table.setItem(row_position, 1, QTableWidgetItem(class_name))
            
            count = class_counts.get(class_id, 0)
            count_item = QTableWidgetItem(str(count))
            
            # Set color based on percentage
            if total_annotations > 0:
                class_percent = (count / total_annotations) * 100
                color = QColor('#4CAF50') if class_percent >= average_percent else QColor('#F44336')
            else:
                color = QColor('white')
            count_item.setForeground(color)
            self.class_table.setItem(row_position, 2, count_item)

    def update_class_dropdown(self):
        self.class_dropdown.clear()
        for class_id, class_name in sorted(self.classes, key=lambda x: x[0]):
            self.class_dropdown.addItem(f"{class_id}: {class_name}", userData=class_id)

    def load_image_directory(self):
        self.image_dir = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        self.load_classes_from_csv()
        if self.image_dir:
            self.image_list = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.current_image_index = 0
            self.update_image_list_widget()
            self.refresh_class_table()
            self.load_image()

    def load_image(self):
        if 0 <= self.current_image_index < len(self.image_list):
            image_path = os.path.join(self.image_dir, self.image_list[self.current_image_index])
            self.current_image = cv2.imread(image_path)
            if self.current_image is not None:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2RGBA)
                self.bounding_boxes = []
                self.load_annotations(image_path)
                self.image_list_widget.setCurrentRow(self.current_image_index)
                self.update_image_display()

    def load_annotations(self, image_path):
        label_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                self.bounding_boxes = [
                    tuple(map(float, line.strip().split())) for line in f.readlines()
                ]
        self.update_annotation_table()

    def update_annotations(self, original_classes, new_classes, deleted_class_id=-1):
        temp_map = {class_name: class_id for class_id, class_name in new_classes}
        class_id_map = {i: temp_map[original_classes[i][1]] for i in range(len(original_classes)) if i != deleted_class_id}
        
        self.update_annotation_table()
        for idx in range(len(self.image_list)):
            image_name = os.path.splitext(self.image_list[idx])[0]
            label_path = os.path.join(self.image_dir, f"{image_name}.txt")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    original_annotations = [
                        tuple(map(float, line.strip().split())) for line in f.readlines()
                    ]
                new_annotations = [(class_id_map[class_id], x_center, y_center, width, height) for class_id, x_center, y_center, width, height in original_annotations]
                with open(label_path, "w") as f:
                    for bbox in new_annotations:
                        class_id, x_center, y_center, width, height = bbox
                        f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        self.load_image()

        print(f"Annotations updated for all images.")

    def update_image_display(self):
        if self.current_image is not None:
            height, width, _ = self.current_image.shape
            aspect_ratio = width / height

            label_height = self.image_label.height()            
            new_width = int(label_height * aspect_ratio)
            new_height = label_height

            self.image_label.resize(new_width, new_height)
            resized_image = cv2.resize(self.current_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            qt_image = QImage(
                resized_image.data, new_width, new_height, resized_image.strides[0], QImage.Format.Format_RGBA8888
            )

            pixmap = QPixmap.fromImage(qt_image)
            painter = QPainter(pixmap)
            pen = QPen(Qt.GlobalColor.red)
            pen.setWidth(2)
            painter.setPen(pen)

            for idx, bbox in enumerate(self.bounding_boxes+self.temp_bounding_box):
                class_id, x_center, y_center, width, height = bbox
                class_id = int(class_id)
                x = int((x_center - width / 2) * pixmap.width())
                y = int((y_center - height / 2) * pixmap.height())
                w = int(width * pixmap.width())
                h = int(height * pixmap.height())

                # Highlight the selected annotation
                if idx in self.selected_rows:
                    pen.setColor(QColor(0, 0, 255))  # Blue for selected annotation
                else:
                    pen.setColor(Qt.GlobalColor.red)  # Red for other annotations
                painter.setPen(pen)

                painter.drawRect(x, y, w, h)
                painter.drawText(int(x), int(y-5), f'{class_id}: {self.classes[class_id][1]}')

            painter.end()
            self.image_label.setPixmap(pixmap)

    def start_drawing(self, event):
        class_id = self.class_search_input.property("selected_class_id")
        if class_id is None:
            QMessageBox.warning(self, "No Class Selected", "Please select a class from the dropdown.")
            return
        self.drawing = True
        self.start_point = event.pos()

    def update_drawing(self, event):
        if self.drawing:
            self.end_point = event.pos()

            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())

            x_center = (x1 + x2) / 2 / self.image_label.width()
            y_center = (y1 + y2) / 2 / self.image_label.height()
            width = (x2 - x1) / self.image_label.width()
            height = (y2 - y1) / self.image_label.height()

            class_id = self.class_search_input.property("selected_class_id")
            self.temp_bounding_box = [(class_id, x_center, y_center, width, height)]

            self.update_image_display()

    def finish_drawing(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.drawing = False
            self.temp_bounding_box = []

            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())

            x_center = (x1 + x2) / 2 / self.image_label.width()
            y_center = (y1 + y2) / 2 / self.image_label.height()
            width = (x2 - x1) / self.image_label.width()
            height = (y2 - y1) / self.image_label.height()

            class_id = self.class_search_input.property("selected_class_id")
            self.bounding_boxes.append((class_id, x_center, y_center, width, height))
            self.bounding_boxes.sort()
            self.update_image_display()
            self.update_annotation_table()

    def save_annotations(self):
        if self.current_image_index >= 0 and self.current_image_index < len(self.image_list):
            image_name = os.path.splitext(self.image_list[self.current_image_index])[0]
            label_path = os.path.join(self.image_dir, f"{image_name}.txt")

            with open(label_path, "w") as f:
                for bbox in self.bounding_boxes:
                    class_id, x_center, y_center, width, height = bbox
                    f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            print(f"Annotations saved to {label_path}")
            self.refresh_class_table()

    def load_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def load_next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()

def main():
    app = QApplication(sys.argv)
    tool = LabelingTool()
    tool.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()