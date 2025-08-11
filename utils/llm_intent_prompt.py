def intent_classifier_prompt(history, query):
    return f"""
Classify the user's message into one of the following categories:
- general_info: Greetings, small talk, off-topic, or general questions not related to the library.
- library_info: Questions about library services, events, locations, policies, opening hours, etc.
- book_search: User is searching for a book by title, topic, or author.
- book_recommend: User wants book recommendations.
- book_lookup_isbn: User provides an ISBN or asks to find a book by ISBN.

[History]: {history}
[User Message]: {query}

Respond with ONLY the category name.
"""