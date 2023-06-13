const express = require('express');
const bodyParser = require('body-parser');
const { promisify } = require('util');
const { PythonShell } = require('python-shell');
const path = require('path');

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

const runPythonShell = promisify(PythonShell.run);

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/predict', async (req, res) => {
  try {
    const comment = req.body.comment;
    console.log(comment);

    const options = {
      args: [comment],
    };

    const result = await runPythonShell('sample.py', options);
    const sentiment = result[0];
    console.log(sentiment)

    console.log(sentiment);
    res.json({ sentiment });
  } catch (err) {
    console.log(err);
    res.status(500).json({ error: 'An error occurred during prediction.' });
  }
});

const port = 3000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
