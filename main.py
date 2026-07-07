from voice import speak, listen
from ai import AIAgent
from config import ASSISTANT_NAME, USER_NAME


def main():
    agent = AIAgent()
    speak(f"Hello {USER_NAME}. I am {ASSISTANT_NAME}.")

    while True:
        command = listen()
        if not command:
            continue

        normalized = command.strip().lower()
        if normalized in {"exit", "quit", "goodbye", "stop"}:
            speak(f"Goodbye {USER_NAME}.")
            break

        reply = agent.ask(command)
        print(f"{ASSISTANT_NAME}: {reply}")
        speak(reply)


if __name__ == "__main__":
    main()
