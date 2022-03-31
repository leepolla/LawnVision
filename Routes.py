import Coloring as Coloring
from flask import Flask, request, jsonify, send_file
app = Flask(__name__)

@app.route("/calculateLawnPercentage", methods=['POST'])
def calculateLawnPercentage():
    postedJson = request.get_json()
    lawnPercentage = Coloring.lawnPercentage(postedJson['geometryPoints'], postedJson['addressHash'])
    data = {
        'lawnPercentage': lawnPercentage
    }
    return jsonify(data)

@app.route("/address<addressHash>/boundless", methods=['GET'])
def getOriginal(addressHash):
    return send_file('data/address%s/boundless.png' % addressHash)

@app.route("/address<addressHash>/plotMap", methods=['GET'])
def getPlot(addressHash):
    return send_file('data/address%s/plotMap.png' % addressHash)

@app.route("/address<addressHash>/greenMap", methods=['GET'])
def getGreen(addressHash):
    return send_file('data/address%s/greenMap.png' % addressHash)

@app.route("/address<addressHash>/filled", methods=['GET'])
def getFilled(addressHash):
    return send_file('data/address%s/filled.png' % addressHash)

@app.route("/address<addressHash>/percent", methods=['GET'])
def getPercentage(addressHash):
    return Coloring.getPercent(addressHash)

@app.route("/clearCache", methods = ['POST'])
def clearCache():
    return Coloring.clearCache()

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')