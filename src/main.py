from rag import answer_question


def main():
    print("RAG system created")
    print("Press 0 to exit")

    while True:
        query = input("You: ")
        if query == "0":
            break

        response, _ = answer_question(query)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    main()
