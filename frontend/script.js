document.addEventListener('DOMContentLoaded', () => {
    const predictButton = document.getElementById('predictButton');
    const reviewTextarea = document.getElementById('reviewText');
    const resultArea = document.getElementById('result');
    const predictionResultDiv = document.getElementById('predictionResult');

    // Hide the result area initially
    resultArea.style.display = 'none';

    predictButton.addEventListener('click', async () => {
        const review = reviewTextarea.value.trim();

        if (review !== "") {
            try {
                const response = await fetch('https://sentiment-analysis-on-movie-reviews-dle9.onrender.com/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ review: review })
                });

                if (response.ok) {
                    const data = await response.json();
                    resultArea.style.display = 'flex'; // Show result area

                    // Display the sentiment from backend
                    predictionResultDiv.textContent = `Sentiment: ${data.sentiment}`;
                    
                    // Set colors based on sentiment
                    if (data.sentiment.toLowerCase().includes('positive')) {
                        predictionResultDiv.style.color = 'green';  // Positive sentiment color
                    } else if (data.sentiment.toLowerCase().includes('negative')) {
                        predictionResultDiv.style.color = 'red';    // Negative sentiment color
                    } else {
                        predictionResultDiv.style.color = 'black';  // For other sentiments
                    }
                } else {
                    resultArea.style.display = 'flex';
                    predictionResultDiv.textContent = 'Error predicting sentiment.';
                    predictionResultDiv.style.color = 'black';
                }
            } catch (error) {
                console.error('Error sending request:', error);
                resultArea.style.display = 'flex';
                predictionResultDiv.textContent = 'Failed to connect to the backend.';
                predictionResultDiv.style.color = 'black';
            }
        } else {
            resultArea.style.display = 'flex';
            predictionResultDiv.textContent = 'Please enter a movie review.';
            predictionResultDiv.style.color = 'black';
        }
    });
});
