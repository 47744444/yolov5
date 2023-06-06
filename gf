import datetime
import pyttsx3
import speech_recognition as sr

tts = pyttsx3.init()
rate = tts.getProperty('rate')
tts.setProperty('rate', rate - 40)
volume = tts.getProperty('volume')
tts.setProperty('volume', volume + 0.9)
voices = tts.getProperty('voices')
tts.setProperty('voice', 'zh-CN')
for voice in voices:
    if voice.name == 'Anna':
        tts.setProperty('voice', voice.id)

rooms = {
    '101': {'booked': False, 'check_in': None, 'check_out': None},
    '102': {'booked': False, 'check_in': None, 'check_out': None},
    '103': {'booked': False, 'check_in': None, 'check_out': None},
    '201': {'booked': False, 'check_in': None, 'check_out': None},
    '202': {'booked': False, 'check_in': None, 'check_out': None},
    '203': {'booked': False, 'check_in': None, 'check_out': None},
    '301': {'booked': False, 'check_in': None, 'check_out': None},
    '302': {'booked': False, 'check_in': None, 'check_out': None},
    '303': {'booked': False, 'check_in': None, 'check_out': None},
}

def book_room(date, room_number):
    if room_number in rooms and not rooms[room_number]['booked']:
        rooms[room_number]['booked'] = True
        rooms[room_number]['check_in'] = get_current_time()
        rooms[room_number]['check_out'] = None  # Initialize check-out time as None
        return f'已成功預訂房間 {room_number}，預定日期為 {date}'
    else:
        return f'房間 {room_number} 不可預訂或不存在'

def talk(word):
    tts.say(word)
    tts.runAndWait()

def get_current_time():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

def get_current_date():
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    return current_date

def check_in(room_number):
    if room_number in rooms and rooms[room_number]['booked']:
        rooms[room_number]['check_in'] = get_current_time()
        return f'已完成入住，入住時間為 {get_current_date()} {rooms[room_number]["check_in"]}'
    else:
        return f'房間 {room_number} 未被預訂或不存在'

def check_out(room_number):
    if room_number in rooms and rooms[room_number]['booked']:
        rooms[room_number]['check_out'] = get_current_time()
        return f'已完成退房，退房時間為 {get_current_date()} {rooms[room_number]["check_out"]}'
    else:
        return f'房間 {room_number} 未被預訂或不存在'

def ask_for_assistance():
    print('還需要我繼續幫忙嗎？')
    talk('還需要我繼續幫忙嗎？')

def chat():
    print('嗨！我是聊天 AI。有什麼我可以幫助你的嗎？')
    talk('嗨！我是聊天 AI。有什麼我可以幫助你的嗎？')
    while True:
        user_input = input('> ')
        if user_input.lower() in ['bye', '再見']:
            print('再見！')
            talk('再見！')
            break

        if '時間' in user_input:
            current_time = get_current_time()
            print(f'目前的時間是 {current_time}')
            talk(f'目前的時間是 {current_time}')
            ask_for_assistance()
        elif '日期' in user_input:
            current_date = get_current_date()
            print(f'今天的日期是 {current_date}')
            talk(f'今天的日期是 {current_date}')
            ask_for_assistance()
        elif '預訂房間' in user_input:
            print('請問要預約哪個日期？')
            talk('請問要預約哪個日期？')
            user_input = input('> ')
            room_input = input('> ')
            booking_date = user_input
            booking_time = get_current_time()
            print(book_room(booking_date, room_input))
            ask_for_assistance()
        elif '入住' in user_input:
            print('請問要入住哪間房間？')
            talk('請問要入住哪間房間？')
            user_input = input('> ')
            print(check_in(user_input))
            ask_for_assistance()
        elif '退房' in user_input:
            print('請問要退房哪間房間？')
            talk('請問要退房哪間房間？')
            user_input = input('> ')
            print(check_out(user_input))
            ask_for_assistance()
        else:
            print('抱歉，我無法理解你的輸入。')
            talk('抱歉，我無法理解你的輸入。')
            ask_for_assistance()

# 執行聊天
chat()
