document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
  
    form.addEventListener('submit', (event) => {
      event.preventDefault();
  
      const commentInput = document.querySelector('#comment');
      const comment = commentInput.value;
  
      // Make a POST request to the server with the comment data
      fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ comment }),
      })
      .then(response => response.json())
      .then(data => {
        // Handle the response and update the UI accordingly
        const sentiment = data.sentiment;
        showResult(sentiment);
      })
      .catch(error => {
        console.error('Error:', error);
      });
    });
  
    function showResult(sentiment) {
      const resultDiv = document.querySelector('#result');
      resultDiv.innerHTML = `The sentiment of the comment is: <strong>${sentiment}</strong>`;
    }
  });
  