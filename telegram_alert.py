# === telegram_alert.py ===
import cv2
import requests
import threading

BOT_TOKEN = 'your_bot_token'
CHAT_ID = 'your_chat_id'
alert_sent = False
alert_lock = threading.Lock() 

def send_photo_async(image):
    global alert_sent
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    _, img_encoded = cv2.imencode('.jpg', image)

    files = {'photo': ('fall.jpg', img_encoded.tobytes())}
    data = {'chat_id': CHAT_ID, 'caption': 'Fall Detection!'}

    try:
        requests.post(url, files=files, data=data)
    except Exception as e:
        print(f"Telegram Error: {e}")

def send_telegram_alert(image):
    global alert_sent
    with alert_lock:
        if alert_sent:
            return
        alert_sent = True

    threading.Thread(target=send_photo_async, args=(image,), daemon=True).start()

def reset_alert():
    global alert_sent
    with alert_lock:
        alert_sent = False
