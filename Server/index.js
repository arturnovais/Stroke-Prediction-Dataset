const express = require('express');

const app = express();

 
const path = require('path')

// então, criamos uma rota para '/'
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname + '/pages/index.html'))
})

//app.use(express.static('public'));
app.use(express.static(path.join(__dirname, 'public')));


const port = process.env.PORT || 3000;

app.listen(port, () => console.log(`Server running on ${port}, http://localhost:${port}`));