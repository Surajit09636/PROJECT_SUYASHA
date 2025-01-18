import speech_recognition as sr
import os
import subprocess
from datetime import datetime



def speech_to_text():
    """Converts speech input to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening for your command...")
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I could not understand that."
        except sr.RequestError:
            return "Sorry, there was an issue with the recognition service."
        except Exception as e:
            return f"Error: {e}"



def save_to_notepad(user_input, ai_response):
    """
    Logs the conversation to a file and opens it in Notepad.
    """
    log_file = "conversation_log.txt"
    with open(log_file, "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp} - You: {user_input}\n")
        file.write(f"{timestamp} - AI: {ai_response}\n\n")


    subprocess.Popen(["notepad.exe", log_file])



def process_command(command):
    """
    Executes user commands and returns appropriate AI responses.
    """
    command = command.lower()

    if "open file" in command:

        try:
            os.startfile("example.txt")
            return "I opened the file as requested."
        except FileNotFoundError:
            return "The file you requested could not be found."
    elif "list files" in command:

        files = os.listdir(".")
        return f"Here are the files: {', '.join(files)}"
    elif "delete file" in command:
        try:
            os.remove("example.txt")
            return "I deleted the file as requested."
        except FileNotFoundError:
            return "The file you requested to delete does not exist."
    elif "time" in command:

        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    elif "exit" in command:
        # End the session
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I didn't understand that command."



def ai_assistant():
    """
    The main AI assistant loop to handle user input and respond accordingly.
    """
    print("AI Assistant is ready. Say 'exit' to end the session.")
    while True:

        user_input = speech_to_text()
        print(f"You: {user_input}")


        ai_response = process_command(user_input)


        print(f"AI: {ai_response}")


        save_to_notepad(user_input, ai_response)


        if "exit" in user_input.lower():
            break



if __name__ == "__main__":
    ai_assistant()
