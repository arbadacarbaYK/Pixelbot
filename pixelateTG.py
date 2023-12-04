import os
from telegram import Update
from telegram.ext import Updater, MessageHandler, CallbackContext, Defaults
from concurrent.futures import ThreadPoolExecutor

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
MAX_THREADS = 5

def start(update: Update, context: CallbackContext):
  update.message.reply_text('Send photo')

def pixelate_faces(update: Update, context: CallbackContext):

  if update.message.photo:
    
    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id) 
    photo_path = f"downloads/{file.file_path}"
    file.download(photo_path)

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
      future = executor.submit(process_image, photo_path, update.message.chat_id, context.bot)  
      
      future.add_done_callback(lambda fut: context.bot.send_photo(fut.result()))

  else:  
    update.message.reply_text('Please send a photo')

# rest of logic 

def main():
  # handler additions

if __name__ == '__main__':
  main()
