import os
import requests

LNbits_API_URL = os.getenv('LNBITS_API_URL')
LNbits_API_KEY = os.getenv('LNBITS_API_KEY')
TIPBOT_API_URL = os.getenv('TIPBOT_API_URL')

def generate_invoice(amount):
    headers = {
        'X-Api-Key': LNbits_API_KEY,
        'Content-Type': 'application/json'
    }
    data = {
        'out': False,
        'amount': amount,
        'memo': 'PixelateBot payment',
        'webhook': '',
    }
    response = requests.post(f'{LNbits_API_URL}/api/v1/payments', json=data, headers=headers)
    return response.json()

def pay_invoice(invoice_id):
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'invoice_id': invoice_id,
    }
    response = requests.post(f'{TIPBOT_API_URL}/pay', json=data, headers=headers)
    return response.json().get('status') == 'paid'

def check_invoice_status(invoice_id):
    headers = {
        'X-Api-Key': LNbits_API_KEY
    }
    response = requests.get(f'{LNbits_API_URL}/api/v1/payments/{invoice_id}', headers=headers)
    return response.json().get('paid')
