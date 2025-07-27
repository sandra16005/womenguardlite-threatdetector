from twilio.rest import Client

def send_sms_alert(message, recipient_number):
    # Replace with your Twilio credentials
    account_sid = 'sid'
    auth_token = 'token'
    twilio_number = 'no'
    
    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=twilio_number,
        to=recipient_number
    )
    print(f"SMS sent to {recipient_number}")
