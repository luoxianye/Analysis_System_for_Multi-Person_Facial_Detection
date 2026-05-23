# smoother.py
from collections import Counter, deque


class EmotionSmoother:
    """
    简单表情平滑器。
    适合视频或摄像头模式，降低单帧误判带来的状态闪烁。
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, emotion: str) -> str:
        self.history.append(emotion)
        counter = Counter(self.history)
        return counter.most_common(1)[0][0]

    def clear(self) -> None:
        self.history.clear()
