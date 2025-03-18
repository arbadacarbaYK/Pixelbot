# Pixelbot 
A Telegram bot that replaces faces with pixels, skull of satoshi, liotta faces, clowns or random cat-pics and can run via github actions or in a local VM.

# https://t.me/Pixelatebot  


<img width="300" alt="Screenshot 2023-12-07 at 13 48 39" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/d896d365-0c91-4c49-890d-eacf8f4143f0">
<img width="300" alt="Screenshot 2023-12-07 at 13 48 12" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/cdbb336a-a351-4553-91bf-687fb4b7c63d">
<img width="300" alt="Screenshot 2023-12-07 at 13 48 05" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/50def6b9-a0d5-4ad4-ab77-064dccd7fe71">
<img width="300" alt="photo_2024-05-24_22-47-22" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/5d0eeab2-dad0-46d1-91a6-b5b4d7ab7b44">
<img width="300" alt="Screenshot 2023-12-07 at 12 48 05" src=https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/1d9a3f17-f87e-4cf7-9cf6-8541bb4bc0e5">
<img width="300" alt="Screenshot 2023-12-07 at 12 48 05" src=https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/7b7ad461-dd32-418f-9da4-f66426d1284c">
<img width="300" alt="Screenshot 2023-12-07 at 12 48 05" src=https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/c8460426-f31e-4968-957c-269ecb970a54)">


## Behaviour in a DM with the bot
*    on first button-press the original pic disappears
*    a keyboard with buttons pops up 
*    choose an option to get an obfuscated version
*    press those buttons until happy with the result
*    works with both static images (JPG/PNG) and animated GIFs

## Behaviour in groupchats where the bot is added
*    bot does not detect faces or react to pics
*    unless any user replies to a pic with the command /pixel
*    the keyboard with buttons pops up and every user can use them
*    cancel option removes the buttons (and leaves pic untouched)
*    works with both static images and animated GIFs - just reply with /pixel to any media

  <img width="250" alt="Screenshot 2024-05-26 at 06 20 45" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/e0885c62-e212-45c0-8c5e-5969f7263351">
<img width="250" alt="Screenshot 2024-05-26 at 06 20 57" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/f4320b49-af19-4805-b909-205887d2cd85">


# Installation on a Server

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Steps to Install

1. Clone the repository or download the source code:
   ```
   git clone https://github.com/arbadacarbayk/Pixelbot.git
   cd Pixelbot
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Copy the example-env a create and .env file including your own Telegram bot token


5. Create required directories:
   ```
   mkdir -p processed downloads 
   ```

6. Run the bot:
   ```
   python pixelateTG.py
   ```

## Running as a Service (Linux)

To keep the bot running after you close the terminal, you can set it up as a systemd service:

1. Create a service file:
   ```
   sudo nano /etc/systemd/system/pixelbot.service
   ```

2. Add the following content (adjust paths as needed):
   ```
   [Unit]
   Description=Pixelbot Telegram Bot
   After=network.target

   [Service]
   User=yourusername
   WorkingDirectory=/path/to/Pixelbot
   ExecStart=/path/to/Pixelbot/venv/bin/python /path/to/Pixelbot/pixelateTG.py
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```
   sudo systemctl enable pixelbot
   sudo systemctl start pixelbot
   ```

4. Check status:
   ```
   sudo systemctl status pixelbot
   ```


<img width="450" alt="Screenshot 2023-12-05 at 20 18 28" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/dcf2e9da-3a32-4371-8454-9d8062bc00f4">

It will then self-build up all dependencies given in requirements.txt and run the bot :) Enjoy.

<img width="450" alt="Screenshot 2023-12-05 at 20 18 36" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/244eeca9-d84e-4ce3-bf68-0dc050fbaa04">

<img width="450" alt="Screenshot 2023-12-05 at 20 18 47" src="https://github.com/arbadacarbaYK/Pixelbot/assets/63317640/0985eec3-58c4-4837-80fd-762368e10b3f">


To Do
- catch more faces
- support more input formats
- moooooaaar masks


⚡️tip me if you like this bot [here](https://lnurlpay.com/LNURL1DP68GURN8GHJ7CN5D9CZUMNV9UH8WETVDSKKKMN0WAHZ7MRWW4EXCUP0X9UXGDEEXQ6XVVM9XUMXGDFCXY6NQS43TRV)
