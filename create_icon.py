from PIL import Image, ImageDraw, ImageFont
import os


def create_ozone_icon():
    """Создание иконки для приложения OSO Forecasting"""

    # Создаем изображение 256x256
    size = (256, 256)
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Фон - градиент от синего к голубому
    for y in range(size[1]):
        for x in range(size[0]):
            # Градиент от #1e3a8a к #0ea5e9
            r = int(30 + (14 - 30) * y / size[1])
            g = int(58 + (165 - 58) * y / size[1])
            b = int(138 + (233 - 138) * y / size[1])
            draw.point((x, y), fill=(r, g, b, 255))

    # Рисуем земной шар
    earth_center = (128, 128)
    earth_radius = 80
    draw.ellipse([earth_center[0] - earth_radius, earth_center[1] - earth_radius,
                  earth_center[0] + earth_radius, earth_center[1] + earth_radius],
                 fill=(79, 70, 229), outline=(255, 255, 255, 200), width=3)

    # Континенты (упрощенно)
    continents = [
        # Северная Америка
        [(100, 80), (120, 70), (140, 80), (130, 100), (110, 95)],
        # Европа
        [(150, 90), (160, 85), (165, 95), (155, 105)],
        # Азия
        [(160, 80), (180, 70), (190, 90), (170, 110), (155, 100)],
        # Южная Америка
        [(120, 120), (130, 130), (125, 160), (115, 150)],
        # Африка
        [(140, 110), (155, 100), (165, 130), (150, 150), (135, 140)],
        # Австралия
        [(180, 150), (190, 145), (195, 160), (185, 165)]
    ]

    for continent in continents:
        draw.polygon(continent, fill=(34, 197, 94))

    # Озоновый слой
    ozone_radius = earth_radius + 15
    draw.ellipse([earth_center[0] - ozone_radius, earth_center[1] - ozone_radius,
                  earth_center[0] + ozone_radius, earth_center[1] + ozone_radius],
                 outline=(34, 211, 238, 180), width=4)

    # Звезды
    stars = [(50, 50), (200, 40), (60, 180), (220, 200), (40, 220)]
    for star in stars:
        draw.ellipse([star[0] - 2, star[1] - 2, star[0] + 2, star[1] + 2],
                     fill=(255, 255, 255))

    # Сохраняем в разных форматах
    if not os.path.exists('assets'):
        os.makedirs('assets')

    image.save('assets/icon.png', 'PNG')

    # Конвертируем в ICO
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    image.save('assets/icon.ico', 'ICO', sizes=ico_sizes)

    print("✅ Иконки созданы: assets/icon.png и assets/icon.ico")


if __name__ == "__main__":
    create_ozone_icon()