def intent_classifier_prompt(history, query):
    return f"""
Classify the user's message into one of the following categories. 
Rules are STRICT, follow exactly:

- general_info: Greetings, small talk, off-topic, or general questions not related to the library.
- library_info: Questions about library services, events, locations, policies, opening hours, etc.
- book_search: The user explicitly wants to FIND, LOOK UP, SEARCH, or LOCATE books by title, topic, or author. 
- book_recommend: The user explicitly uses words like RECOMMEND, SUGGEST, GIVE ME A LIST, or similar, asking for recommendations. 
- book_lookup_isbn: The user provides an ISBN or asks to find a book by ISBN.

[History]: {history}
[User Message]: {query}

Respond with ONLY the category name.
"""
