from chatterbot import ChatBot

# 產生一個 ChatBot，加入兩個邏輯連接器(logic adapters)
bot = ChatBot(
    'Math & Time Bot',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter'
    ]
)

# 數學式估算
response = bot.get_response('What is 4 + 9?')
print(response)

# 目前時間查詢
response = bot.get_response('What time is it?')
print(response)
