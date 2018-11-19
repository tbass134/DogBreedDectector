require('dotenv').config()

'use strict';

let express = require('express');
let bodyParser = require('body-parser');
let facebookInterface = require('./facebookInterface');

let app = express();

const inboundFacebookEvent = (event) => {
    console.log('event recieved from fb endpoint: ', event);
}

app.use(bodyParser.json({ type: 'application/json' }));


app.get('/', (req, res, next) => {
    res.sendStatus(200);       
});

app.get('/webhook', function(req, res) {
    return facebookInterface.verifyToken(req,res)
});

app.post('/webhook', function (req, res) {
   return facebookInterface.receivedWebhook(req, res)
});

var port = process.env.port || process.env.PORT || 3000;
app.listen(port, () => { 
    console.log('Chatbot application Running on port ' + port) 
});