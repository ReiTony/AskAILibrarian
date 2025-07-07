import random
from rapidfuzz import process, fuzz

# ------------------------------------------
# Default fallback suggestions
# ------------------------------------------
default_reminders = [
    "Could you clarify your query or provide more details?",
    "Try rephrasing your question for better results.",
    "Explore general information about the library."
]

# ------------------------------------------
# Predefined keyword-based suggestions
# ------------------------------------------
keyword_suggestions = {
    "library": [
        "How can I apply for a Library ID at MSU?",
        "How to obtain a Library ID at MSU?",
        "Which documents are required to apply for a Library ID at MSU?",
        "How can I borrow resources from the library at MSU?",
        "What is the process for borrowing books from the library?"
    ],
    "divisions": [
        "What is the Access Services Division?",
        "What services are offered at American Corner Marawi?",
        "What does the College/Unit Libraries Services Division do?",
        "What services are provided by the Information Technology Services Division?",
        "What services does the Malano UPHub offer?"
    ],
    "services": [
        "What is the role of the Administrative Services Division?",
        "What does the College/Unit Libraries Services Division do?",
        "What does the Information Technology Services Division do?"
    ],
    "library_director": [
        "Who is the University Library Director?",
        "What are the responsibilities of the University Library Director?",
        "Who is Zaalica B. Edres?"
    ],
    "about": [
        "What is the University Archives and Memorabilia?",
        "Where is the University Archives and Memorabilia located?",
        "What does the Access Services Division do?",
        "What is the Malano UPHub?"
    ],
    "division_head": [
        "Who is the head of the College/Unit Libraries Services Division?",
        "Who manages the College/Unit Libraries Services Division?",
        "Who is the head of the University Archives and Memorabilia?",
        "Who is the head of the Access Services Division?",
        "Who manages the Information Technology Services Division?"
    ],
    "staff_and_personnel": [
        "Who works in the Administrative Services Division?",
        "Can I get a list of staff in the Administrative Services Division?",
        "Who are the librarians and staff in the College/Unit Libraries Services Division?",
        "Who works in the College/Unit Libraries Services Division?",
        "Who is in charge of the Metadata (E-Books) section?",
        "Who is responsible for the Metadata (E-Theses & E-Dissertations) section?",
        "Who manages Library ID card printing in the IT Services Division?",
        "Who are the librarians and staff of the University Archives and Memorabilia?",
        "Who works at American Corner Marawi?",
        "Can I get a list of personnel at American Corner Marawi?"
    ],
    "contact": [
        "Where is American Corner Marawi located?",
        "What other American Corner locations are there in Luzon, Visayas, and Mindanao?",
        "How can I contact the Access Services Division?",
        "What is the contact information for the Access Services Division?",
        "How can I contact American Corner Marawi?",
        "Where is American Corner Marawi located in the Main Library?",
        "What is the contact information for the University Library Director?",
        "How can I contact the Malano UPHub?"
    ],
    "email_addresses": [
        "What is the email address of American Corner Marawi?",
        "What is the email address of the Administrative Services Division?",
        "What is the email address of the Access Services Division?",
        "What is the email address of the Information Technology Services Division?"
    ],
    "service_hours": [
        "What are the service hours of the Access Services Division?",
        "What are the service hours of American Corner Marawi?",
        "What are the service hours for the University Library?",
        "Is the University Library open on weekends?",
        "What are the service hours of the MSU University Main Library?",
        "When does the Administrative Services Division close each day?"
    ],
    "mini_conference_room": [
        "How can I reserve the Mini-Conference Room?",
        "What are the booking requirements for the Mini-Conference Room?",
        "What is the maximum capacity of the Mini-Conference Room?",
        "How much does it cost to book the Mini-Conference Room?",
        "What is the cancellation policy for Mini-Conference Room bookings?"
    ],
}

# ------------------------------------------
# Suggestion logic with randomization and variety
# ------------------------------------------
def get_suggestions(user_query, query_keywords, previous_suggestions=None):
    """
    Returns up to 3 unique, varied suggestions based on keyword matching.
    Adds randomization so repeated queries can show different suggestions.
    Avoids repeats within the response and can optionally avoid previously shown suggestions.
    """
    suggestions = set()
    matched_categories = set()

    # Lowercase for fuzzy matching
    for keyword in set(kw.lower() for kw in query_keywords):
        closest_match = process.extractOne(
            keyword,
            keyword_suggestions.keys(),
            scorer=fuzz.partial_ratio
        )
        if closest_match and closest_match[1] > 40:
            matched_keyword = closest_match[0]
            if matched_keyword in matched_categories:
                continue
            matched_categories.add(matched_keyword)
            options = keyword_suggestions[matched_keyword][:]
            random.shuffle(options)
            for option in options[:2]:
                suggestions.add(option)
                if len(suggestions) >= 3:
                    break
        if len(suggestions) >= 3:
            break

    if previous_suggestions:
        suggestions -= set(previous_suggestions)

    out = list(suggestions)
    random.shuffle(out)
    for rem in default_reminders:
        if len(out) >= 3:
            break
        if rem not in out:
            out.append(rem)
    return out[:3]
