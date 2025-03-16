document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("sentiment-form").addEventListener("submit", function (event) {
        event.preventDefault();  // Prevent form submission
        
        let reviewText = document.getElementById("review").value;
        
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ review: reviewText })  // Send JSON data
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerText = `Sentiment: ${data.sentiment} (Confidence: ${data.confidence.toFixed(2)})`;
        })
        .catch(error => console.error("Error:", error));
    });
});

