from flask import Flask, request, render_template_string
from transformers import pipeline
import textstat

# Load summarizer (smaller model to work on Hugging Face free tier)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Writer - Text Summarizer</title>
</head>
<body style="font-family: Arial; margin: 40px;">
    <h2>AI Writer - Text Summarizer</h2>
    <form method="post">
        <textarea name="article" rows="12" cols="80" placeholder="Paste your text here"></textarea><br><br>
        <input type="submit" value="Summarize">
    </form>
    {% if summary %}
        <h3>Summary:</h3>
        <p>{{ summary }}</p>
        <h4>Word Count:</h4>
        <p>{{ word_count }}</p>
        <h4>Readability Score:</h4>
        <p>{{ readability }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    summary, word_count, readability = None, None, None
    if request.method == "POST":
        article = request.form["article"]
        if article.strip():
            # Summarize
            result = summarizer(article, max_length=150, min_length=40, do_sample=False)
            summary = result[0]["summary_text"]

            # Extra stats
            word_count = len(summary.split())
            readability = textstat.flesch_reading_ease(summary)

    return render_template_string(HTML_TEMPLATE, summary=summary,
                                  word_count=word_count,
                                  readability=readability)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
