'use strict';

let fetch = require('node-fetch');
var request = require('request');
var predictions = require('./predictions.js')
require('dotenv').config()

const APP_ID = process.env.MESSENGER_APP_ID
console.log("APP_ID",APP_ID)
const VERIFY_TOKEN = process.env.MESSENGER_VERIFY_TOKEN
const APP_SECRET = process.env.MESSENGER_APP_SECRET;
const PAGE_ACCESS_TOKEN = process.env.MESSENGER_PAGE_ACCESS_TOKEN;
const SERVER_URL = process.env.SERVER_URL

var graphapi = request.defaults({
    baseUrl: 'https://graph.facebook.com/v2.9',
    json: true,
    auth: {
        'bearer': PAGE_ACCESS_TOKEN
    }
});

// Enable page subscriptions for this app, using the app-page token
exports.enableSubscriptions = function () {
    graphapi({
        url: '/me/subscribed_apps',
        method: 'POST'
    }, function (error, response, body) {
        // This should return with {success:true}, otherwise you've got problems!
        console.log('enableSubscriptions', body);
    });
}

exports.subscribeWebhook = function () {
    graphapi({
        url: '/app/subscriptions',
        method: 'POST',
        auth: { 'bearer': APP_ID + '|' + APP_SECRET },
        qs: {
            'object': 'page',
            'fields': 'message_deliveries,messages,messaging_postbacks',
            'verify_token': VERIFY_TOKEN,
            'callback_url': SERVER_URL + '/webhook'
        }
    }, function (error, response) {
        if (error) {
            console.log(error);
        } else {
            console.log('subscribeWebhook', response.body);
        }
    });
}

/*
 * Call the Send API. The message data goes in the body. If successful, we'll 
 * get the message id in a response 
 */
function callSendAPI(messageData) {
    let qs = "?access_token=" + PAGE_ACCESS_TOKEN;
    return fetch('https://graph.facebook.com/v2.6/me/messages' + qs, {
        method: 'POST',
        body: JSON.stringify(messageData),
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    })
        .then(function (res) {
            return res.json();
        })
        .catch(err => {
            console.log("Failed calling Send API", err); /// show status code, status message and error
        })
}

/*
 * Verify that the callback came from Facebook. Using the App Secret from
 * your custom integration, we can verify the signature that is sent with each
 * callback in the x-hub-signature field, located in the header.
 *
 * https://developers.facebook.com/docs/graph-api/webhooks#setup
 *
 */

exports.verifyRequestSignature = function (req, res, buf) {
    var signature = req.headers['x-hub-signature'];

    if (!signature) {
        // For testing, let's log an error. In production, you should throw an
        // error.
        console.log('Couldn\'t validate the signature.');
    } else {
        var elements = signature.split('=');
        var signatureHash = elements[1];

        var expectedHash = crypto.createHmac('sha1', APP_SECRET)
            .update(buf)
            .digest('hex');

        if (signatureHash != expectedHash) {
            throw new Error('Couldn\'t validate the request signature.');
        }
    }
}

exports.verifyToken = function (req, res) {

    if (req.query['hub.mode'] === 'subscribe' &&
        req.query['hub.verify_token'] === VERIFY_TOKEN) {
        console.log('Validating webhook');
        res.status(200).send(req.query['hub.challenge']);
    } else {
        console.log('Failed validation. Make sure the validation tokens match.');
        res.sendStatus(403);
    }
}

exports.receivedWebhook = function (req, res) {
    try {
        var data = req.body;

        // Make sure this is a page subscription
        if (data.object == 'page') {
            // Iterate over each entry
            // There may be multiple if batched
            data.entry.forEach(function (pageEntry) {
                // Iterate over each messaging event
                pageEntry.changes.forEach(function (messagingEvent) {
                    if (messagingEvent.field == "mention") {
                        var message = messagingEvent.value.message.toLowerCase()
                        //If any text is sent to the bot, we'll get the image url from the post,
                        //make a prediction on our Flask API,
                        //get the breed name,
                        //and post a comment into that post with the name of the breed dectected
                        exports.getPostContents(messagingEvent.value.post_id)
                        .then(function (photo_url) {
                            return predictions.predictBreed(photo_url)
                        }).then(function (breed_name) {
                            exports.addCommentAPI(messagingEvent.value.post_id, breed_name)
                        }).catch(function (error) {
                            exports.addCommentAPI(messagingEvent.value.post_id, "Sorry, but I'm unable to tell you what dog that is")
                        })
                        
                    }
                    else {
                        console.log('Webhook received unknown messagingEvent: ', messagingEvent);
                    }
                });
            });
        }
    } catch (err) {
        console.log("receivedWebhook caught an error: " + error)
    }

    // Assume all went well.
    // You must send back a 200, within 20 seconds, to let us know you've
    // successfully received the callback. Otherwise, the  res.sendStatus(200); request will time out.
    res.sendStatus(200);
}


/*
 * Call the Send API. The message data goes in the body. If successful, we'll
 * get the message id in a response
 *
 */
function callSendAPI(messageData) {
    graphapi({
        url: '/me/messages',
        method: 'POST',
        json: messageData
    }, function (error, response, body) {
        if (!error && response.statusCode == 200) {
            var recipientId = body.recipient_id;
            var messageId = body.message_id;

            if (messageId) {
                console.log('Successfully sent message with id %s to recipient %s',
                    messageId, recipientId);
            } else {
                console.log('Successfully called Send API for recipient %s',
                    recipientId);
            }
        } else {
            console.log('Failed calling Send API', response.statusCode, response.statusMessage, body.error);
        }
    });
}
/**
 * Post a comment into the current thread (postId)
 */
exports.addCommentAPI = function (postId, message) {
    return new Promise(function (resolve, reject) {
        var commentURL = '/' + postId + "/comments?message=" + message
        console.log('info', 'posting comment with url: ', commentURL)
        graphapi({
            url: commentURL,
            method: 'POST'
        }, function (error, response, body) {
            if (!error && response.statusCode == 200) {
                var recipientId = body.recipient_id;
                var messageId = body.message_id;

                if (messageId) {
                    console.log('Successfully sent message with id %s to recipient %s',
                        messageId, recipientId);
                    resolve()
                } else {
                    console.log('Successfully called Add Comment API for recipient %s',
                        recipientId);
                    reject()
                }
            } else {
                console.log('Failed calling Send API', response.statusCode, response.statusMessage, body.error);
                reject()
            }
        });

    })
}
/**
 * Get the first attachment from the current thread (postId)
 */
exports.getPostContents = function (postId) {
    return new Promise(function (resolve, reject) {

        var commentURL = '/' + postId + "/attachments"
        console.log('info', 'posting comment with url: ', commentURL)
        graphapi({
            url: commentURL,
            method: 'GET'
        }, function (error, response, body) {
            console.log("getPostContents", body)
            if (!error && response.statusCode == 200) {
                var photo = body.data[0].media.image.src;
                if (photo) {
                    resolve(photo)
                }
            } else {
                console.log('Failed calling Send API', response.statusCode, response.statusMessage, body.error);
                reject()
            }
        });
    })
}