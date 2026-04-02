import ollama

def response_from_llm(context, query):
    prompt = f"""
        You are a helpful assistant.

        Answer ONLY using the provided context.

        STRICT RULES:
        - Do NOT use outside knowledge
        - If answer not found, say "I don't know"
        - Keep everything concise
        - Follow the exact structure below
        - Do NOT add extra sections
        - STRICTLY follow word limits
        - Do NOT write paragraphs beyond limits

        FORMAT:

        Introduction:
        - 1–2 short sentences only

        Key Points:
        - Maximum 5 bullet points
        - No long explanations

        Summary:
        - 1 short sentence

        Next Steps (if applicable):
        - 1–2 bullet points (optional)

        Context:
        {context}

        Question:
        {query}

        Answer:
    """
    response_text = ""

    stream = ollama.chat(
        model='mistral',
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        token = chunk['message']['content']
        response_text += token
        print(token, end='', flush=True)