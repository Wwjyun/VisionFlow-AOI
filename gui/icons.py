from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPainterPath, QPen, QPixmap

# ============================================================
# AOI Console — line-icon set (24px viewBox, stroke ~1.7, round caps)
# Simplified QPainter re-implementation of design_handoff_aoi_gui/app/icons.jsx
# ============================================================


def _painter(pixmap: QPixmap, color: str, stroke: float) -> QPainter:
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(color)
    pen.setWidthF(stroke)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    return painter


def _draw(name: str, painter: QPainter, scale: float) -> None:
    s = scale

    def pt(x: float, y: float) -> QPointF:
        return QPointF(x * s, y * s)

    def rect(x: float, y: float, w: float, h: float) -> QRectF:
        return QRectF(x * s, y * s, w * s, h * s)

    if name == "play":
        path = QPainterPath()
        path.moveTo(pt(7, 5))
        path.lineTo(pt(19, 12))
        path.lineTo(pt(7, 19))
        path.closeSubpath()
        painter.drawPath(path)

    elif name == "image":
        painter.drawRoundedRect(rect(3, 4, 18, 16), 2 * s, 2 * s)
        painter.drawEllipse(pt(9, 10), 1.6 * s, 1.6 * s)
        path = QPainterPath()
        path.moveTo(pt(5, 18))
        path.lineTo(pt(10, 13))
        path.lineTo(pt(13, 16))
        path.lineTo(pt(16, 13))
        path.lineTo(pt(19, 16))
        painter.drawPath(path)

    elif name == "recipe":
        path = QPainterPath()
        path.moveTo(pt(7, 3))
        path.lineTo(pt(15, 3))
        path.lineTo(pt(19, 7))
        path.lineTo(pt(19, 21))
        path.lineTo(pt(7, 21))
        path.closeSubpath()
        painter.drawPath(path)
        path2 = QPainterPath()
        path2.moveTo(pt(15, 3))
        path2.lineTo(pt(15, 7))
        path2.lineTo(pt(19, 7))
        painter.drawPath(path2)
        painter.drawLine(pt(9, 12), pt(15, 12))
        painter.drawLine(pt(9, 16), pt(15, 16))

    elif name == "designer":
        path = QPainterPath()
        path.moveTo(pt(4, 20))
        path.lineTo(pt(8, 19))
        path.lineTo(pt(19, 8))
        path.lineTo(pt(16, 5))
        path.lineTo(pt(5, 16))
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(pt(14, 7), pt(17, 10))

    elif name == "table":
        painter.drawRoundedRect(rect(3, 4, 18, 16), 2 * s, 2 * s)
        painter.drawLine(pt(3, 10), pt(21, 10))
        painter.drawLine(pt(9, 10), pt(9, 20))
        painter.drawLine(pt(15, 10), pt(15, 20))

    elif name == "bar_chart":
        painter.drawLine(pt(4, 20), pt(20, 20))
        painter.drawLine(pt(5, 20), pt(5, 5))
        painter.drawRoundedRect(rect(8, 12, 2.8, 6), 0.8 * s, 0.8 * s)
        painter.drawRoundedRect(rect(12, 8, 2.8, 10), 0.8 * s, 0.8 * s)
        painter.drawRoundedRect(rect(16, 4, 2.8, 14), 0.8 * s, 0.8 * s)

    elif name == "gear":
        painter.drawEllipse(pt(12, 12), 3 * s, 3 * s)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            painter.drawLine(pt(12 + dx * 9.5, 12 + dy * 9.5), pt(12 + dx * 6.5, 12 + dy * 6.5))
        for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            painter.drawLine(pt(12 + dx * 6.7, 12 + dy * 6.7), pt(12 + dx * 4.6, 12 + dy * 4.6))

    elif name == "folder":
        path = QPainterPath()
        path.moveTo(pt(3, 7))
        path.cubicTo(pt(3, 5.9), pt(3.9, 5), pt(5, 5))
        path.lineTo(pt(9, 5))
        path.lineTo(pt(11, 7))
        path.lineTo(pt(19, 7))
        path.cubicTo(pt(20.1, 7), pt(21, 7.9), pt(21, 9))
        path.lineTo(pt(21, 18))
        path.cubicTo(pt(21, 19.1), pt(20.1, 20), pt(19, 20))
        path.lineTo(pt(5, 20))
        path.cubicTo(pt(3.9, 20), pt(3, 19.1), pt(3, 18))
        path.closeSubpath()
        painter.drawPath(path)

    elif name == "check":
        path = QPainterPath()
        path.moveTo(pt(4, 12.5))
        path.lineTo(pt(9, 17.5))
        path.lineTo(pt(20, 6.5))
        painter.drawPath(path)

    elif name == "x":
        painter.drawLine(pt(6, 6), pt(18, 18))
        painter.drawLine(pt(18, 6), pt(6, 18))

    elif name == "chevron_right":
        path = QPainterPath()
        path.moveTo(pt(9, 6))
        path.lineTo(pt(15, 12))
        path.lineTo(pt(9, 18))
        painter.drawPath(path)

    elif name == "chevron_down":
        path = QPainterPath()
        path.moveTo(pt(6, 9))
        path.lineTo(pt(12, 15))
        path.lineTo(pt(18, 9))
        painter.drawPath(path)

    elif name in ("zoom_in", "zoom_out"):
        painter.drawEllipse(pt(11, 11), 7 * s, 7 * s)
        painter.drawLine(pt(16.2, 16.2), pt(21, 21))
        painter.drawLine(pt(8, 11), pt(14, 11))
        if name == "zoom_in":
            painter.drawLine(pt(11, 8), pt(11, 14))

    elif name == "fit":
        for x, y, dx, dy in ((9, 4, -1, 0), (15, 4, 1, 0), (9, 20, -1, 0), (15, 20, 1, 0)):
            path = QPainterPath()
            if dy == 0 and y == 4:
                path.moveTo(pt(x, y))
                path.lineTo(pt(x + dx * 5, y))
                path.lineTo(pt(x + dx * 5, y + 5))
            else:
                path.moveTo(pt(x, y))
                path.lineTo(pt(x + dx * 5, y))
                path.lineTo(pt(x + dx * 5, y - 5))
            painter.drawPath(path)

    elif name == "save":
        path = QPainterPath()
        path.moveTo(pt(5, 3))
        path.lineTo(pt(16, 3))
        path.lineTo(pt(21, 8))
        path.lineTo(pt(21, 19))
        path.cubicTo(pt(21, 20.1), pt(20.1, 21), pt(19, 21))
        path.lineTo(pt(5, 21))
        path.cubicTo(pt(3.9, 21), pt(3, 20.1), pt(3, 19))
        path.lineTo(pt(3, 5))
        path.cubicTo(pt(3, 3.9), pt(3.9, 3), pt(5, 3))
        painter.drawPath(path)
        path2 = QPainterPath()
        path2.moveTo(pt(8, 3))
        path2.lineTo(pt(8, 8))
        path2.lineTo(pt(15, 8))
        path2.lineTo(pt(15, 3))
        painter.drawPath(path2)
        path3 = QPainterPath()
        path3.moveTo(pt(7, 21))
        path3.lineTo(pt(7, 14))
        path3.lineTo(pt(17, 14))
        path3.lineTo(pt(17, 21))
        painter.drawPath(path3)

    elif name == "eye":
        path = QPainterPath()
        path.moveTo(pt(2.5, 12))
        path.cubicTo(pt(2.5, 12), pt(6, 5.5), pt(12, 5.5))
        path.cubicTo(pt(18, 5.5), pt(21.5, 12), pt(21.5, 12))
        path.cubicTo(pt(21.5, 12), pt(18, 18.5), pt(12, 18.5))
        path.cubicTo(pt(6, 18.5), pt(2.5, 12), pt(2.5, 12))
        painter.drawPath(path)
        painter.drawEllipse(pt(12, 12), 2.8 * s, 2.8 * s)

    elif name == "crosshair":
        painter.drawEllipse(pt(12, 12), 7 * s, 7 * s)
        painter.drawLine(pt(12, 2), pt(12, 6))
        painter.drawLine(pt(12, 18), pt(12, 22))
        painter.drawLine(pt(2, 12), pt(6, 12))
        painter.drawLine(pt(18, 12), pt(22, 12))

    elif name == "camera":
        path = QPainterPath()
        path.moveTo(pt(3, 9))
        path.cubicTo(pt(3, 7.9), pt(3.9, 7), pt(5, 7))
        path.lineTo(pt(7.5, 7))
        path.lineTo(pt(9, 4.5))
        path.lineTo(pt(15, 4.5))
        path.lineTo(pt(16.5, 7))
        path.lineTo(pt(19, 7))
        path.cubicTo(pt(20.1, 7), pt(21, 7.9), pt(21, 9))
        path.lineTo(pt(21, 17))
        path.cubicTo(pt(21, 18.1), pt(20.1, 19), pt(19, 19))
        path.lineTo(pt(5, 19))
        path.cubicTo(pt(3.9, 19), pt(3, 18.1), pt(3, 17))
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawEllipse(pt(12, 13), 3.5 * s, 3.5 * s)

    elif name == "history":
        painter.drawEllipse(rect(3.5, 3.5, 17, 17))
        painter.drawLine(pt(12, 8), pt(12, 12.5))
        painter.drawLine(pt(12, 12.5), pt(15, 14.5))


def icon(name: str, size: int = 18, color: str = "#51616a", stroke: float = 1.7) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = _painter(pixmap, color, stroke * (size / 24.0))
    _draw(name, painter, size / 24.0)
    painter.end()
    return QIcon(pixmap)


def pixmap(name: str, size: int = 18, color: str = "#51616a", stroke: float = 1.7) -> QPixmap:
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    painter = _painter(px, color, stroke * (size / 24.0))
    _draw(name, painter, size / 24.0)
    painter.end()
    return px
