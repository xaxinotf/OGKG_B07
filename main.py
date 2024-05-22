import sys
import numpy as np
import cvxpy as cp
from scipy.spatial import Voronoi
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QPolygonF, QBrush, QPen
from PyQt5.QtCore import Qt, QPointF


class StarPolygonEllipseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.points = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Star-Shaped Polygon and Inscribed Ellipse Generator')
        self.setGeometry(100, 100, 1920, 1080)  # Розмір вікна 1920x1080

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        layout = QVBoxLayout(self.main_widget)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        self.generate_button = QPushButton('Generate')
        self.generate_button.clicked.connect(self.generate)
        layout.addWidget(self.generate_button)

        self.view.viewport().installEventFilter(self)

        self.show()

    def eventFilter(self, source, event):
        if event.type() == event.MouseButtonPress and source is self.view.viewport():
            self.add_point(event.pos())
        return super().eventFilter(source, event)

    def add_point(self, pos):
        if len(self.points) >= 5000:
            return
        point = self.view.mapToScene(pos)
        self.points.append((point.x(), point.y()))
        self.scene.addEllipse(point.x() - 2, point.y() - 2, 4, 4, QPen(), QBrush(Qt.green))

    def generate(self):
        if len(self.points) < 6:
            print("Потрібно хоча б 6 точок для побудови зіркового многокутника.")
            return

        self.scene.clear()

        # Підготовка точок для многокутника
        N = len(self.points)
        K = 2  # Крок зірковості

        points = np.array(self.points)
        center = np.mean(points, axis=0)

        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_points = points[np.argsort(angles)]

        # Створення зіркового многокутника
        star_points = []
        for i in range(N):
            star_points.append(sorted_points[i])
            star_points.append(sorted_points[(i + K) % N])

        star_points = star_points[:N]  # Тільки N точок

        px, py = zip(*star_points)
        px = np.array(px)
        py = np.array(py)

        # Створення QPolygonF для зіркового многокутника
        polygon = QPolygonF([QPointF(x, y) for x, y in zip(px, py)])
        self.scene.addPolygon(polygon, QPen(Qt.green))

        # Обчислення вписаного еліпса
        m = len(px)
        pxint = np.mean(px)
        pyint = np.mean(py)

        A = np.zeros((m, 2))
        b = np.zeros(m)

        for i in range(m):
            edge = np.array([px[(i + 1) % m] - px[i], py[(i + 1) % m] - py[i]])
            normal = np.array([-edge[1], edge[0]])
            A[i, :] = normal / np.linalg.norm(normal)
            b[i] = A[i, :].dot(0.5 * (np.array([px[i], py[i]]) + np.array([px[(i + 1) % m], py[(i + 1) % m]])))
            if A[i, :].dot(np.array([pxint, pyint])) - b[i] > 0:
                A[i, :] = -A[i, :]
                b[i] = -b[i]

        B = cp.Variable((2, 2), symmetric=True)
        d = cp.Variable(2)

        constraints = [cp.norm(B @ A[i, :], 2) + A[i, :] @ d <= b[i] for i in range(m)]
        objective = cp.Maximize(cp.log_det(B))

        prob = cp.Problem(objective, constraints)
        prob.solve()

        B_value = B.value
        d_value = d.value

        angles = np.linspace(0, 2 * np.pi, 200)
        ellipse_points = np.dot(B_value, np.array([np.cos(angles), np.sin(angles)])) + d_value[:, np.newaxis]

        ellipse_poly = QPolygonF()
        for x, y in zip(ellipse_points[0], ellipse_points[1]):
            ellipse_poly.append(QPointF(x, y))
        self.scene.addPolygon(ellipse_poly, QPen(Qt.red, 2, Qt.DashLine))

        for point in self.points:
            self.scene.addEllipse(point[0] - 2, point[1] - 2, 4, 4, QPen(), QBrush(Qt.green))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = StarPolygonEllipseApp()
    sys.exit(app.exec_())
