/**
 * Makes a request to the Flask server on heroku to preform the prediction.
 * Accepts `image_path` which is the path to the image from workplace
 */
var request = require("request");

exports.predictBreed = function (image_path) {
    return new Promise(function (resolve, reject) {

    var options = {
        method: 'POST',
        url: process.env.PREDICTON_SERVER_URL+'/predict',
        body: { image_path: image_path },
        json: true
    };

    request(options, function (error, response, body) {
        if (error) {reject(error); return};
        console.log(body);
        resolve(body["prediction"])
    });
})

}