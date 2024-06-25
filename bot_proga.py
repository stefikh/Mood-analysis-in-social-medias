import telebot
import pickle
import random
from telebot import types
import numpy as np
import pandas as pd
import re
import string

#Модель
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

#Результат векторизации с коэффицентами
feature_names = np.load('feature_names.npy', allow_pickle=True)
coefficients = model.coef_[0]
all_words = {word: 'Positif' if coef > 12 else 'Négatif' for word, coef in zip(feature_names, coefficients) if abs(coef) > 12}

#Векторизатор
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

#Бот
token = ''
bot = telebot.TeleBot(token)

#Предобработка текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation.replace("'", "")), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
    
#Важные слова для сообщения пользователя
def extract_important_words(model, vectorizer, text_vector, prob, len_text):
    contributions = text_vector.multiply(model.coef_[0])
    feat_names = vectorizer.get_feature_names_out()
    contributions_df = pd.DataFrame({'Feature': feat_names, 'Contribution': contributions.toarray()[0]})
    contributions_df = contributions_df.sort_values(by='Contribution', ascending=False)
    x_num = 2
    if len_text <= 3:
        x_num = 1
    if prob < 0.5:
        return list(contributions_df.tail(x_num)['Feature'])
    return list(contributions_df.head(x_num)['Feature'])

#Оценка текста
def analyze_sentiment(model, vectorizer, text):
    preprocessed_text = preprocess_text(text)
    text_vector = vectorizer.transform([preprocessed_text])
    probability = model.predict_proba(text_vector)[:, 1]
    len_text = len(text.split())
    important_words = extract_important_words(model, vectorizer, text_vector, probability[0], len_text)
    if probability[0] >= 0.75:
        return 'Какое прекрасное письмо! Эти слова точно понравились бы Королеве, да, даже самой Констанции! Однозначно положительное письмо из-за слов: ' + ', '.join(important_words)
    elif 0.75 > probability[0] >= 0.5:
        return 'Друг мой, если Вы будете использовать такой стиль при переписке, то ваш собеседник может немного расстроиться. Смягчите чуть-чуть ваш тон. Скорее положительный текст из-за: ' + ', '.join(important_words)
    elif 0.5 > probability[0] >= 0.25:
        return 'Каналья! Это на грани допустимого, королеве такое точно нельзя говорить. В Вашем сообщении есть всё-таки отрицательные слова, но не прямо серьёзные: ' + ', '.join(important_words)
    else:
        return 'Друг мой, вы чего?! Это же настоящее оскорбление или такое сильное недовольство, что даже я так не ругаюсь. Ваше сообщение пропитано отрицательными словами. Например: ' + ', '.join(important_words)

#Приветственное сообщение
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, (
        "Тысяча чертей!  Позвольте представиться, бот — Д'Артаньян! Готов помочь вам с изучением тональности слов на французском языке."
    ))

#Что может бот
@bot.message_handler(commands=['help'])
def help(message):
    helpani = '''Что я умею?
• Определять тональность сообщения;
• Выделять слова, которые влияют на тональность;
• Освежить в памяти слова, которые уже всплывали в предыдущих определениях тональности.'''
    bot.send_message(message.chat.id, helpani)

#Повтор слов
@bot.message_handler(commands=['game'])
def game_start(message):
    game_loop(message.chat.id)

def game_loop(chat_id):
    game_word = random.choice(list(all_words.keys()))

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Positif", callback_data=f'{game_word} Positif'))
    markup.add(types.InlineKeyboardButton("Négatif", callback_data=f'{game_word} Négatif'))

    #Отправляем кнопочки
    bot.send_message(chat_id, f"Как вы считаете, друг, слово '{game_word}' positif ou négatif?",
                     reply_markup=markup)

#Кнопочки для игры
@bot.callback_query_handler(func=lambda call: call.data in ['continue', 'stop'])
def continue_stop_query(call):
    if call.data == 'continue':
        #Продолжаем игру
        game_loop(call.message.chat.id)
    else:
        #Заканчиваем игру
        bot.send_message(call.message.chat.id, "Отличная вышла игра, друг мой! Теперь можно отправляться хоть на службу к королеве")

#Когда пользователь нажал на кнопочку
@bot.callback_query_handler(func=lambda call: call.data.split()[1] in ['Positif', 'Négatif'])
def callback_query(call):
    word, answer = call.data.split()

    if answer == all_words[word]:
        bot.send_message(call.message.chat.id, "Вы угадали! Скоро Вам станут известны все тайны Лувра!")
    else:
        bot.send_message(call.message.chat.id, f"Куда вас сударь к черту занесло? Это неправильно! Правильный ответ: {all_words[word]}")

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Продолжить игру", callback_data='continue'))
    markup.add(types.InlineKeyboardButton("Остановить игру", callback_data='stop'))
    bot.send_message(call.message.chat.id, "Продолжим нашу игру? Готов поставить 5 экю, что Вы ещё и не такое можете!", reply_markup=markup)

@bot.message_handler(commands=['stop'])
def finish(message):
    bot.send_message(message.chat.id, 'Мне грустно с Вами прощаться. Надеюсь, что скоро всё-таки отправимся в приключение!')

#Результат оценки
@bot.message_handler(func=lambda message: True)
def analyze_message(message):
    bot.reply_to(message, "Дайте подумать, друг мой...")
    response = analyze_sentiment(model, vectorizer, message.text)
    bot.reply_to(message, response)

bot.polling(none_stop=True, interval=0)