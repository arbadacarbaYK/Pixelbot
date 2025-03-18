from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_main_keyboard(session_id=None):
    keyboard = [
        [
            InlineKeyboardButton("⚔️ Pixelate", callback_data=f"{session_id}:pixelate" if session_id else "pixelate"),
            InlineKeyboardButton("✂️ Full Pixelate", callback_data=f"{session_id}:full_pixelate" if session_id else "full_pixelate")
        ],
        [
            InlineKeyboardButton("🤡 Clown", callback_data=f"{session_id}:clown" if session_id else "clown"),
            InlineKeyboardButton("😎 Ray Liotta", callback_data=f"{session_id}:liotta" if session_id else "liotta")
        ],
        [
            InlineKeyboardButton("💀 Skull", callback_data=f"{session_id}:skull" if session_id else "skull"),
            InlineKeyboardButton("😺 Cat", callback_data=f"{session_id}:cat" if session_id else "cat"),
            InlineKeyboardButton("🐸 Pepe", callback_data=f"{session_id}:pepe" if session_id else "pepe")
        ],
        [
            InlineKeyboardButton("👨 Chad", callback_data=f"{session_id}:chad" if session_id else "chad")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_pixelation_keyboard(session_id=None):
    keyboard = [
        [InlineKeyboardButton("Very Fine", callback_data=f"{session_id}:pixelate_very_fine" if session_id else "pixelate_very_fine"),
         InlineKeyboardButton("Fine", callback_data=f"{session_id}:pixelate_fine" if session_id else "pixelate_fine")],
        [InlineKeyboardButton("Rough", callback_data=f"{session_id}:pixelate_rough" if session_id else "pixelate_rough"),
         InlineKeyboardButton("Very Rough", callback_data=f"{session_id}:pixelate_very_rough" if session_id else "pixelate_very_rough")],
        [InlineKeyboardButton("« Back", callback_data=f"{session_id}:back" if session_id else "back")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_full_pixelation_keyboard(session_id=None):
    keyboard = [
        [InlineKeyboardButton("Very Fine", callback_data=f"{session_id}:full_0.2" if session_id else "full_0.2"),
         InlineKeyboardButton("Fine", callback_data=f"{session_id}:full_0.15" if session_id else "full_0.15")],
        [InlineKeyboardButton("Rough", callback_data=f"{session_id}:full_0.09" if session_id else "full_0.09"),
         InlineKeyboardButton("Very Rough", callback_data=f"{session_id}:full_0.05" if session_id else "full_0.05")],
        [InlineKeyboardButton("« Back", callback_data=f"{session_id}:back" if session_id else "back")]
    ]
    return InlineKeyboardMarkup(keyboard) 