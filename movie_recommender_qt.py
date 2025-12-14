import sys
import ast
import requests
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QHBoxLayout, QScroller, QScrollerProperties, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= OMDb CONFIG =================
OMDB_API_KEY = "ba2a0382"

# ================= DATA PROCESSING =================
movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

movies = movies.merge(credits, on="title")
movies = movies[["title", "overview", "genres", "keywords", "cast", "crew"]]

def convert(text):
    try:
        return [i["name"] for i in ast.literal_eval(text)]
    except:
        return []

def get_cast(text):
    try:
        return [i["name"] for i in ast.literal_eval(text)[:3]]
    except:
        return []

def get_director(text):
    try:
        for i in ast.literal_eval(text):
            if i["job"] == "Director":
                return i["name"]
    except:
        pass
    return ""

movies["overview"] = movies["overview"].fillna("")
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(get_cast)
movies["crew"] = movies["crew"].apply(get_director)

movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["crew"]
)

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

# ================= FETCH POSTER =================
def fetch_poster(title):
    try:
        res = requests.get(
            "http://www.omdbapi.com/",
            params={"apikey": OMDB_API_KEY, "t": title},
            timeout=5
        ).json()
        poster = res.get("Poster")
        if poster and poster != "N/A":
            return poster
    except:
        pass
    return None

# ================= LOADING SCREEN =================
class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(520, 320)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet("background-color:#0f1117;")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Movie Recommender")
        title.setFont(QFont("Segoe UI Semibold", 28))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#f9fafb; letter-spacing:1.4px;")

        subtitle = QLabel("Warning: This May Cause Binge-Watching")
        subtitle.setFont(QFont("Segoe UI", 13))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:#9ca3af;")

        self.dots = QLabel("Loading")
        self.dots.setFont(QFont("Segoe UI", 14))
        self.dots.setAlignment(Qt.AlignCenter)
        self.dots.setStyleSheet("color:#6366f1;")

        self.progress = QProgressBar()
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                background:#1a1d24;
                border-radius:3px;
            }
            QProgressBar::chunk {
                background:#6366f1;
                border-radius:3px;
            }
        """)

        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addWidget(self.dots)
        layout.addSpacing(14)
        layout.addWidget(self.progress)
        layout.addStretch()

        self.dot_count = 0
        self.value = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(120)

    def animate(self):
        self.dot_count = (self.dot_count + 1) % 4
        self.dots.setText("Loading" + "." * self.dot_count)

        self.value += 3
        self.progress.setValue(self.value)

# ================= MOVIE CARD =================
class MovieCard(QWidget):
    def __init__(self, title, poster_url):
        super().__init__()
        self.setStyleSheet("background:#1a1d24; border-radius:16px;")

        layout = QHBoxLayout(self)
        layout.setSpacing(16)

        poster = QLabel()
        poster.setFixedSize(100, 150)

        if poster_url:
            pixmap = QPixmap()
            pixmap.loadFromData(requests.get(poster_url).content)
            poster.setPixmap(
                pixmap.scaled(
                    100, 150,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        else:
            poster.setStyleSheet("background:#2f3342; border-radius:8px;")

        title_label = QLabel(title)
        title_label.setWordWrap(True)
        title_label.setFont(QFont("Segoe UI Semibold", 14))
        title_label.setStyleSheet("color:#f9fafb;")

        layout.addWidget(poster)
        layout.addWidget(title_label)

# ================= MAIN APP =================
class MovieRecommender(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Movie Recommender")
        self.setFixedSize(760, 640)
        self.setStyleSheet("background-color:#0f1117;")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Movie Recommender")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI Semibold", 26))
        title.setStyleSheet("color:#f9fafb;")

        subtitle = QLabel("Warning: This May Cause Binge-Watching")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 13))
        subtitle.setStyleSheet("color:#9ca3af;")

        self.input = QLineEdit()
        self.input.setPlaceholderText("Search movies, actors, directors…")
        self.input.returnPressed.connect(self.recommend)
        self.input.setFont(QFont("Segoe UI", 14))
        self.input.setStyleSheet("""
            QLineEdit {
                background:#1a1d24;
                color:#ffffff;
                padding:16px;
                border-radius:14px;
                border:1px solid #2f3342;
            }
        """)

        button = QPushButton("Recommend")
        button.clicked.connect(self.recommend)
        button.setFont(QFont("Segoe UI Semibold", 14))
        button.setStyleSheet("""
            QPushButton {
                background:#4f46e5;
                color:white;
                padding:14px;
                border-radius:22px;
            }
            QPushButton:hover {
                background:#6366f1;
            }
        """)

        self.listbox = QListWidget()
        self.listbox.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.listbox.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listbox.setStyleSheet("background:transparent; border:none;")

        scroller = QScroller.scroller(self.listbox.viewport())
        QScroller.grabGesture(self.listbox.viewport(), QScroller.LeftMouseButtonGesture)

        props = scroller.scrollerProperties()
        props.setScrollMetric(QScrollerProperties.DecelerationFactor, 0.05)
        props.setScrollMetric(QScrollerProperties.MaximumVelocity, 0.6)
        scroller.setScrollerProperties(props)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(16)
        layout.addWidget(self.input)
        layout.addWidget(button)
        layout.addSpacing(12)
        layout.addWidget(self.listbox)
        self.setLayout(layout)

    def recommend(self):
        query = self.input.text().strip().lower()
        self.listbox.clear()

        if not query:
            return

        index = None
        for i in range(len(movies)):
            if query in movies.iloc[i]["title"].lower():
                index = i
                break

        if index is None:
            return

        scores = sorted(
            enumerate(similarity[index]),
            key=lambda x: x[1],
            reverse=True
        )

        for i in scores[1:6]:
            title = movies.iloc[i[0]].title
            poster_url = fetch_poster(title)

            item = QListWidgetItem()
            card = MovieCard(title, poster_url)
            item.setSizeHint(card.sizeHint())

            self.listbox.addItem(item)
            self.listbox.setItemWidget(item, card)

# ================= RUN =================
app = QApplication(sys.argv)

loading = LoadingScreen()
loading.show()

window = MovieRecommender()

def start_app():
    loading.close()
    window.show()

QTimer.singleShot(3000, start_app)

sys.exit(app.exec_())
