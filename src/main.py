from rag import answer_question
from uuid import uuid4


def main():
    print("RAG system created")
    print("Press 0 to exit")
    session_id = str(uuid4())

    while True:
        query = input("You: ")
        if query == "0":
            break

        response, _ = answer_question(query, session_id=session_id)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    main()
