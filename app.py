from flask import Flask, render_template

app = Flask(__name__) # creates flask object and links to this module

@app.route("/") # call this function whenever user visits the root (/) page
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(port = 5000, debug = True)